"""Exact readable menu/reference invariants; run through the Slurm check job."""
from copy import deepcopy
import hashlib
import json

import pytest

from app.models import OperationOptions, VerificationReport
from graph_local.backends import LocalHFRoles
from graph_local.menu_records import prepare_menu_request
from graph_local.profile import BenchmarkProfile
from graph_local.specialist_context import (
    grouped_menu_actions, reconstruct_menu_actions, resolve_specialist_receipt,
    readable_payload_json, resolve_specialist_contract, resolve_specialist_action_counts)
from test_backends import FakeHF
from test_specialist_context import payload, generated


def request(source=None, role='diagnostic', protocol='menu-ids-v3'):
    return prepare_menu_request(role, source if source is not None else payload(),
        OperationOptions if role == 'diagnostic' else VerificationReport, protocol=protocol)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def test_every_literal_action_reconstructs_and_binds_without_filtering_or_ranking():
    source = payload()
    commands = source['benchmark_public_context']['complete_offered_commands_and_presets']
    extras = [
        {'kind': 'invoke', 'instrument_id': 'furnace', 'capability_id': 'menu.configure',
         'arguments': {'enabled': False, 'n': 1, 'temperature': 1.0, 'nested': [None, {'a': 'b'}]}},
        {'kind': 'invoke', 'instrument_id': 'furnace', 'capability_id': 'menu.configure',
         'arguments': {'enabled': True, 'n': 1.0, 'temperature': 1, 'nested': []}},
        {'kind': 'wait', 'dt_s': 1.0}, {'kind': 'wait', 'dt_s': 1},
        {'kind': 'finish'}]
    commands.extend({'command': {'text': 'Exact action'},
        'argument_presets': [{'text': canonical(action.get('arguments', action)), 'action': action}]}
        for action in extras)
    result = request(source)
    assert canonical(reconstruct_menu_actions(result.payload['offered_action_groups'])) == canonical(result.actions)
    assert result.actions == request(source, protocol='menu-ids-v2').actions
    for identifier, action in result.actions.items():
        record = result.schema.model_validate(generated(identifier)).model_dump()
        bound = result.bind(record)['candidates'][0]['intent']['arguments']['action']
        assert canonical(bound) == canonical(action)
    assert result.audit['ranked_or_filtered_actions'] is False
    assert result.audit['menu_action_count'] == len(result.actions)


def test_only_identical_prefix_fields_are_shared_and_all_other_native_fields_remain_literal():
    menu = [{'menu_action_id': 'm' + str(index), 'action': action, 'preset_text': 'Helpful preset'}
        for index, action in enumerate([
            {'kind': 'invoke', 'instrument_id': 'ika', 'capability_id': 'x', 'arguments': {'a': 1}},
            {'kind': 'invoke', 'instrument_id': 'ika', 'capability_id': 'x', 'arguments': {'a': 2}},
            {'kind': 'invoke', 'instrument_id': None, 'arguments': {}},
            {'kind': 'invoke', 'arguments': {}},
            {'kind': 'read', 'instrument_id': 'ika', 'state_ids': ['a', 'b']}])]
    groups, _ = grouped_menu_actions(menu)
    assert groups[0]['common_action'] == {'kind': 'invoke', 'instrument_id': 'ika', 'capability_id': 'x'}
    assert [item['arguments'] for item in groups[0]['choices']] == [{'a': 1}, {'a': 2}]
    assert len(groups) == 4  # A missing instrument is not a null instrument.
    assert canonical(reconstruct_menu_actions(groups)) == canonical({item['menu_action_id']: item['action'] for item in menu})


def test_preset_text_is_omitted_only_for_exact_json_redundancy():
    action = {'kind': 'invoke', 'instrument_id': 'ika', 'capability_id': 'x', 'arguments': {'a': 1.0}}
    menu = [{'menu_action_id': 'm' + str(index), 'action': deepcopy(action), 'preset_text': text}
        for index, text in enumerate(['{"a":1.0}', '  {"a": 1.0} ', '{"a":1}',
                                      'Set a carefully', '{"a":NaN}', None])]
    groups, omitted = grouped_menu_actions(menu)
    assert omitted == ['m0', 'm1']
    choices = groups[0]['choices']
    assert 'preset_text' not in choices[0]
    assert [item['preset_text'] for item in choices[2:]] == ['{"a":1}', 'Set a carefully', '{"a":NaN}', None]
    assert len(reconstruct_menu_actions(groups)) == len(menu)


def test_exact_mechanically_derived_read_wait_text_is_omitted_without_changing_actions():
    cases = [
        ({'kind': 'read', 'instrument_id': 'ika', 'state_ids': ['a', 'b']}, 'Read a, b', True),
        ({'kind': 'read', 'state_ids': ['a', 'b']}, 'Read a, b for safety', False),
        ({'kind': 'read', 'state_ids': ['b', 'a']}, 'Read a, b', False),
        ({'kind': 'read', 'state_ids': None}, 'Read all native observations', False),
        ({'kind': 'wait', 'dt_s': 30.0}, 'Wait 30 simulated seconds', True),
        ({'kind': 'wait', 'dt_s': 0.5}, 'Wait 0.5 simulated seconds', True),
        ({'kind': 'wait', 'dt_s': 30}, 'Wait 30 simulated seconds for settling', False),
        ({'kind': 'wait', 'dt_s': True}, 'Wait 1 simulated seconds', False)]
    menu = [{'menu_action_id': 'm' + str(index), 'action': action, 'preset_text': text}
            for index, (action, text, omitted) in enumerate(cases)]
    audit = {}
    groups, json_omitted = grouped_menu_actions(menu, omission_audit=audit)
    choices = {choice['menu_action_id']: choice for group in groups for choice in group['choices']}
    assert not json_omitted
    assert audit['derived_preset_text_omitted_ids'] == ['m0', 'm4', 'm5']
    for index, (action, text, omitted) in enumerate(cases):
        choice = choices['m' + str(index)]
        assert ('preset_text' not in choice) == omitted
        if not omitted:
            assert choice['preset_text'] == text
    assert canonical(reconstruct_menu_actions(groups)) == canonical({item['menu_action_id']: item['action'] for item in menu})


def test_exact_cached_observation_map_is_shown_once_and_references_resolve():
    source = payload()
    cache = source['benchmark_public_context']['current_public_view']['observation']['observations']
    result = request(source)
    old = request(source, protocol='menu-ids-v2')
    history = result.payload['native_receipts_in_execution_order']
    assert history[0]['response']['observations_ref'] == 'cached_native_observation.observations'
    assert 'observations' not in history[0]['response']
    assert canonical(result.payload).count(canonical(cache)) == 1
    for new_receipt, old_receipt in zip(history, old.payload['native_receipts_in_execution_order']):
        assert canonical(resolve_specialist_receipt(result.payload, new_receipt)) == canonical(old_receipt)
    assert result.audit['observation_reference_history_indices'] == [0]


def test_partial_observation_maps_reference_only_exact_cached_state_ids_and_resolve():
    source = payload()
    public = source['benchmark_public_context']
    cache = public['current_public_view']['observation']['observations']
    cache['temperature'] = {'value': 313.15, 'quality': 'valid', 'sample_time_s': 1.0, 'unit': 'K'}
    previous = public['complete_native_history'][0]['response']['observations']
    previous['temperature'] = {'value': 312.15, 'quality': 'valid', 'sample_time_s': 0.0, 'unit': 'K'}
    previous['historical-only'] = {'value': 'retained literal', 'sample_time_s': 0.0}
    before = deepcopy(source)
    value = request(source)
    receipt = value.payload['native_receipts_in_execution_order'][0]
    assert receipt['response']['observations_from_cached'] == ['program.running']
    assert receipt['response']['observations'] == {name: reading for name, reading in previous.items() if name != 'program.running'}
    old = request(source, protocol='menu-ids-v2').payload['native_receipts_in_execution_order'][0]
    assert canonical(resolve_specialist_receipt(value.payload, receipt)) == canonical(old)
    assert value.audit['observation_entry_references'] == [{'history_index': 0, 'state_ids': ['program.running']}]
    assert source == before


@pytest.mark.parametrize('field,value', [('value', 0), ('sample_time_s', 0), ('quality', 'measured')])
def test_partial_references_preserve_differing_types_timestamps_and_quality(field, value):
    source = payload()
    public = source['benchmark_public_context']
    cache = public['current_public_view']['observation']['observations']
    cache['unchanged'] = {'value': {'observations_from_cached': ['literal-native-data']}, 'sample_time_s': 0.0}
    previous = public['complete_native_history'][0]['response']['observations']
    previous['unchanged'] = deepcopy(cache['unchanged'])
    previous['program.running'][field] = value
    result = request(source)
    row = result.payload['native_receipts_in_execution_order'][0]
    assert row['response']['observations_from_cached'] == ['unchanged']
    assert canonical(row['response']['observations']['program.running']) == canonical(previous['program.running'])
    assert canonical(resolve_specialist_receipt(result.payload, row)) == canonical(
        request(source, protocol='menu-ids-v2').payload['native_receipts_in_execution_order'][0])


@pytest.mark.parametrize('field,replacement', [('quality', 'measured'), ('sample_time_s', 0), ('value', 0)])
def test_different_native_quality_time_or_value_type_is_never_replaced(field, replacement):
    source = payload()
    native = source['benchmark_public_context']['complete_native_history'][0]['response']['observations']
    native['program.running'][field] = replacement
    result = request(source)
    receipt = result.payload['native_receipts_in_execution_order'][0]
    assert canonical(receipt['response']['observations']) == canonical(native)
    assert 'observations_ref' not in receipt['response']
    assert result.payload['cached_reading_provenance']['cached_values_equal_that_receipt'] is False


def test_latest_successful_receipt_appears_once_with_indexed_human_readable_history_refs():
    source = payload()
    source['benchmark_public_context']['complete_native_history'].pop()
    result = request(source)
    view = result.payload
    assert view['latest_native_receipt']['response']['result']['running'] is True
    assert view['latest_successful_native_receipt'] == {'history_index': 1, 'receipt_ref': 'latest_native_receipt'}
    assert view['native_receipts_in_execution_order'][1] == {'history_index': 1, 'receipt_ref': 'latest_native_receipt'}
    expected = request(source, protocol='menu-ids-v2').payload['latest_native_receipt']
    assert resolve_specialist_receipt(view, view['latest_successful_native_receipt']) == expected
    assert canonical(view).count('"running":true') == 1


def test_distinct_latest_busy_and_successful_start_are_both_preserved_and_resolvable():
    result = request()
    view = result.payload
    assert view['latest_native_receipt']['response']['error']['code'] == 'BUSY'
    assert view['latest_successful_native_receipt']['response']['result']['running'] is True
    assert [row['history_index'] for row in view['native_receipts_in_execution_order']] == [0, 1, 2]
    for index in (1, 2):
        assert resolve_specialist_receipt(view, view['native_receipts_in_execution_order'][index])['history_index'] == index
    text = readable_payload_json(view)
    assert text.index('"goal"') < text.index('"latest_native_receipt"') < text.index('"cached_native_observation"')


def test_identical_proposal_uses_receipt_local_executed_action_reference():
    source = payload()
    result = request(source)
    old = request(source, protocol='menu-ids-v2')
    assert result.payload['latest_native_receipt']['proposed_action_ref'] == 'executed_action'
    assert 'proposed_action' not in result.payload['latest_native_receipt']
    assert result.audit['proposed_action_reference_history_indices'] == [0, 1, 2]
    for row, original in zip(result.payload['native_receipts_in_execution_order'],
                             old.payload['native_receipts_in_execution_order']):
        assert canonical(resolve_specialist_receipt(result.payload, row)) == canonical(original)


@pytest.mark.parametrize('proposed', [
    {'kind': 'wait', 'dt_s': 1}, {'kind': 'wait', 'dt_s': True},
    {'kind': 'wait', 'dt_s': 2.0}])
def test_different_or_numerically_coerced_proposals_keep_literal_native_action(proposed):
    source = payload()
    history = source['benchmark_public_context']['complete_native_history']
    history[-1]['executed_action'] = {'kind': 'wait', 'dt_s': 1.0}
    history[-1]['proposed_action'] = deepcopy(proposed)
    result = request(source)
    last = result.payload['latest_native_receipt']
    assert 'proposed_action_ref' not in last
    assert canonical(last['proposed_action']) == canonical(proposed)
    assert canonical(last['executed_action']) == canonical(history[-1]['executed_action'])


def test_source_and_frozen_v2_remain_unchanged_and_no_oracle_sections_reappear():
    source = payload()
    original = deepcopy(source)
    v2_before = request(source, protocol='menu-ids-v2')
    result = request(source)
    v2_after = request(source, protocol='menu-ids-v2')
    assert source == original
    assert v2_before.payload == v2_after.payload
    assert v2_before.prompt == v2_after.prompt
    assert v2_before.audit == v2_after.audit
    assert result.audit['source_payload_sha256'] == v2_before.audit['source_payload_sha256']
    expected_contract = deepcopy(v2_before.payload['current_public_contract'])
    for capability in expected_contract['capabilities'].values():
        capability.pop('input_schema', None)
    assert resolve_specialist_contract(result.payload['current_public_contract']) == expected_contract
    assert result.payload['public_manual'] == v2_before.payload['public_manual']
    text = canonical(result.payload)
    assert all(word not in text for word in ('SECRET_ORACLE', 'OBSOLETE_INITIAL_GOAL', 'OBSOLETE_MODEL_PROPOSAL',
        '$public_ref', '$public_table', 'presence_bitmask'))


@pytest.mark.parametrize('role', ['diagnostic', 'verification'])
def test_v3_omits_only_parameter_schemas_and_audits_exact_hash_without_changing_validation(role):
    source = payload()
    original = deepcopy(source)
    value = request(source, role=role)
    old = request(source, role=role, protocol='menu-ids-v2')
    contract = deepcopy(old.payload['current_public_contract'])
    hashes = {}
    for name, capability in contract['capabilities'].items():
        if 'input_schema' in capability:
            schema = capability.pop('input_schema')
            encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            hashes[name] = hashlib.sha256(encoded.encode()).hexdigest()
    assert resolve_specialist_contract(value.payload['current_public_contract']) == contract
    assert value.audit['omitted_input_schema_sha256_by_capability'] == hashes
    assert value.schema.model_json_schema() == old.schema.model_json_schema()
    assert source == original  # Native source schemas remain present and untouched.
    assert 'input_schema' in source['benchmark_public_context']['current_public_view']['contract']['capabilities']['thermal.start_program']
    assert 'native validator is unchanged' in value.prompt
    if role == 'diagnostic':
        assert value.actions == old.actions
        for identifier in value.actions:
            record = value.schema.model_validate(generated(identifier)).model_dump()
            assert canonical(value.bind(record)) == canonical(old.bind(record))


def test_duplicate_precondition_lists_reference_first_exact_list_and_restore_all_conditions():
    source = payload()
    capabilities = source['benchmark_public_context']['current_public_view']['contract']['capabilities']
    conditions = [{'description_zh': 'Literal public condition', 'state_ids': ['power.on'],
                   'check': 'Runtime.guard', 'enforced_by': 'runtime', 'value': 1.0}]
    capabilities['thermal.start_program']['preconditions'] = deepcopy(conditions)
    capabilities['exact-copy'] = {'preconditions': deepcopy(conditions), 'description': 'Distinct command'}
    different = deepcopy(conditions)
    different[0]['value'] = 1
    capabilities['different-type'] = {'preconditions': different}
    before = deepcopy(source)
    result = request(source)
    view = result.payload['current_public_contract']
    assert view['capabilities']['thermal.start_program']['preconditions'] == conditions
    assert view['capabilities']['exact-copy']['preconditions_ref'] == 'thermal.start_program'
    assert 'preconditions' not in view['capabilities']['exact-copy']
    assert view['capabilities']['different-type']['preconditions'] == different
    expected = request(source, protocol='menu-ids-v2').payload['current_public_contract']
    for capability in expected['capabilities'].values():
        capability.pop('input_schema', None)
    assert canonical(resolve_specialist_contract(view)) == canonical(expected)
    assert result.audit['preconditions_reference_by_capability'] == {'exact-copy': 'thermal.start_program'}
    assert source == before


def test_action_counts_reference_only_exact_current_offered_actions_and_restore_original_counts():
    source = payload()
    public = source['benchmark_public_context']
    history = public['complete_native_history']
    history.append({'executed_action': {'kind': 'wait', 'dt_s': 30.0}, 'response': {'status': 'succeeded'}})
    result = request(source)
    rows = result.payload['public_history_progress']['identical_executed_action_counts']
    assert any(row.get('action_from_menu_id') == 'm0' for row in rows)
    # Offered wait has integer30, so the float30.0 history action remains literal.
    assert any(row.get('action') == {'kind': 'wait', 'dt_s': 30.0} for row in rows)
    expected = request(source, protocol='menu-ids-v2').payload['public_history_progress']['identical_executed_action_counts']
    assert canonical(resolve_specialist_action_counts(result.payload)) == canonical(expected)


def test_verifier_has_no_menu_but_keeps_citations_and_empty_history_is_valid():
    source = payload()
    source['benchmark_public_context']['complete_native_history'] = []
    result = request(source, role='verification')
    assert result.payload['latest_native_receipt'] is None
    assert result.payload['latest_successful_native_receipt'] is None
    assert result.payload['native_receipts_in_execution_order'] == []
    assert 'offered_action_groups' not in result.payload
    assert result.schema.model_validate({'status': 'unresolved', 'evidence_ids': ['wrapper-3'], 'missing_evidence': []})
    with pytest.raises(ValueError):
        result.schema.model_validate({'status': 'supported', 'evidence_ids': ['fake'], 'missing_evidence': []})


def test_v3_actual_wire_uses_grouped_actions_records_source_hash_and_bounds_whitespace(monkeypatch):
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
    fake = FakeHF(raw=json.dumps(generated()), generated=50)
    fake.tokenizer = object()
    roles = LocalHFRoles({'record_protocol': 'menu-ids-v3', 'structured_output': 'json_schema_lmfe_v4'}, backend=fake)
    result = roles.invoke('diagnostic', 'Original prompt', payload(), OperationOptions)
    assert captured == {'max_consecutive_whitespaces': 16}
    wire = json.loads(fake.calls[0]['messages'][1]['content'])
    assert wire['protocol'] == 'menu-ids-v3'
    assert reconstruct_menu_actions(wire['offered_action_groups']) == request().actions
    assert result['candidates'][0]['intent']['arguments']['action'] == request().actions['m0']
    assert roles.last_call['source_payload_sha256'] == roles.last_call['menu_binding']['source_payload_sha256']
    assert roles.last_call['record_protocol'] == 'menu-ids-v3'
    assert roles.last_call['usage']['completion_tokens'] == 50


def test_v3_diagnostic_receives_exact_historical_verifier_feedback_without_mutating_state():
    source = payload()
    feedback = {'status': 'unresolved', 'evidence_ids': ['older-wrapper-1'],
        'missing_evidence': ['Current program completion has not been observed.'],
        'evidence_signature': 'original-signature', 'observation_id': 'older-wrapper-1'}
    state = {'target': source['target'], 'observations': source['observations'],
        'capabilities': {}, 'last_error': source['last_error'],
        'last_verification_feedback': deepcopy(feedback)}
    class Lab:
        def context(self):
            return deepcopy(source['benchmark_public_context'])
    projected = BenchmarkProfile(Lab(), record_protocol='menu-ids-v3').projection('diagnostic', state, 'request')
    assert canonical(projected['last_verification_feedback']) == canonical(feedback)
    wire = request(projected).payload
    historical = wire['historical_last_verification_feedback']
    assert canonical(historical['report']) == canonical(feedback)
    assert 'historical' in historical['meaning'] and 'does not assert current or fresh' in historical['meaning']
    # Neither projection retargets old citations to the current wrapper ID.
    assert historical['report']['observation_id'] == 'older-wrapper-1'
    assert historical['report']['evidence_ids'] != ['wrapper-3']
    projected['last_verification_feedback']['missing_evidence'].append('Mutated projected copy')
    assert state['last_verification_feedback'] == feedback
    assert historical['report'] == feedback
    assert 'historical_last_verification_feedback' not in request(projected, role='verification').payload


def test_v3_actual_duplicate_candidate_repair_wire_retains_error_and_previous_response():
    source = payload()
    repeated = generated()
    repeated['candidates'].append(deepcopy(repeated['candidates'][0]))
    fake = FakeHF(raw=json.dumps(repeated), generated=50)
    roles = LocalHFRoles({'record_protocol': 'menu-ids-v3'}, backend=fake)
    with pytest.raises(ValueError, match='distinct currently offered menu IDs') as original_error:
        roles.invoke('diagnostic', 'Original prompt', source, OperationOptions)
    original_wire = json.loads(fake.calls[0]['messages'][1]['content'])
    # This is the graph's actual repair envelope when invoke fails before it
    # returns a bound application record (raw is None in app.graph.call_role).
    repair = {'instruction': 'Fix only the JSON/schema error; return the requested record.',
        'validation_error': str(original_error.value),
        'previous_response': deepcopy(roles.last_call['response'])}
    repair_source = dict(deepcopy(source), format_repair=deepcopy(repair))
    with pytest.raises(ValueError, match='distinct currently offered menu IDs'):
        roles.invoke('diagnostic', 'Original prompt', repair_source, OperationOptions)
    repair_wire = json.loads(fake.calls[1]['messages'][1]['content'])
    assert fake.calls[0]['messages'] != fake.calls[1]['messages']
    assert 'format_repair' not in original_wire
    assert repair_wire['format_repair'] == repair
    assert json.loads(repair_wire['format_repair']['previous_response']['choices'][0]['message']['content']) == repeated
    assert 'not a native observation or an executed action' in repair_wire['format_repair_semantics']
    assert repair_wire['cached_native_observation'] == original_wire['cached_native_observation']
    assert reconstruct_menu_actions(repair_wire['offered_action_groups']) == reconstruct_menu_actions(original_wire['offered_action_groups'])
    repair_wire['format_repair']['previous_response']['choices'][0]['message']['content'] = 'Changed local test copy'
    assert repair_source['format_repair'] == repair
    assert 'format_repair' not in source


def test_v3_preserves_invalid_repair_response_fields_and_leaves_v2_projection_unchanged():
    source = payload()
    repair_source = deepcopy(source)
    # Invalid model fields must survive as quoted prior output; they are not
    # allowed native evidence and must not be interpreted as native state.
    repair_source['format_repair'] = {'instruction': 'Fix only the JSON/schema error; return the requested record.',
        'validation_error': 'Unexpected field in previous model response',
        'previous_response': {'private': 'MODEL_INVENTED_FIELD', 'candidates': []}}
    result = request(repair_source)
    assert result.payload['format_repair'] == repair_source['format_repair']
    assert request(repair_source, protocol='menu-ids-v2').payload == request(source, protocol='menu-ids-v2').payload
    assert request(repair_source, protocol='menu-ids-v2').prompt == request(source, protocol='menu-ids-v2').prompt


@pytest.mark.parametrize('protocol', [None, 'menu-ids-v1', 'menu-ids-v2'])
def test_older_profile_projections_do_not_receive_new_verification_feedback(protocol):
    source = payload()
    state = {'target': source['target'], 'observations': source['observations'],
        'capabilities': {}, 'last_verification_feedback': {'status': 'unresolved', 'missing_evidence': ['Old gap']}}
    class Lab:
        def context(self):
            return deepcopy(source['benchmark_public_context'])
    baseline = BenchmarkProfile(Lab()).projection('diagnostic', state, 'request')
    projected = BenchmarkProfile(Lab(), record_protocol=protocol).projection('diagnostic', state, 'request')
    assert canonical(projected) == canonical(baseline)
    assert 'last_verification_feedback' not in projected
    assert 'historical_last_verification_feedback' not in request(projected, protocol='menu-ids-v2').payload


def test_v3_explains_finish_and_receipt_semantics_without_changing_v1_v2_prompts():
    old_receipt_text = 'latest_successful_native_receipt preserves the last successful command before any later rejection.'
    finish_text = "Returning status 'supported' immediately ends this episode as a FINISH claim for the ENTIRE public goal."
    verifier = request(role='verification')
    assert verifier.prompt.startswith(finish_text)
    assert 'It does not mean an observation is valid' in verifier.prompt
    for role in ('diagnostic', 'verification'):
        newer = request(role=role)
        assert old_receipt_text not in newer.prompt
        assert 'which may be a read, wait or command' in newer.prompt
        assert finish_text not in request(role=role, protocol='menu-ids-v1').prompt
        older = request(role=role, protocol='menu-ids-v2')
        assert old_receipt_text in older.prompt
        assert finish_text not in older.prompt
        assert 'historical_last_verification_feedback' not in older.prompt
    # Wording clarifies the existing interface; every original verdict remains model-selectable.
    for status in ('supported', 'unresolved', 'contradicted'):
        record = verifier.schema.model_validate({'status': status, 'evidence_ids': ['wrapper-3'], 'missing_evidence': []})
        assert verifier.bind(record.model_dump())['status'] == status
