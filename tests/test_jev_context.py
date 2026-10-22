"""Check that smaller Jev requests retain facts and do not alter candidate decisions."""
from copy import deepcopy
import json

import httpx
import pytest

from app.jev_context import compact_context, operation_request, POLICY_VERSION
from app.openrouter_jev import MODEL, OpenRouterJevOperationBackend
from app.operation_decision import validate_decision
from app.token_budget import TokenLedger


def candidate(name='power', cap='device::power.set', args=None):
    return {'candidate_id': name, 'intent': {'capability_id': cap,
        'arguments': {'enabled': False} if args is None else args,
        'discriminating_observation': 'Read actual state.', 'postcondition': 'Verify the requested state.'},
        'rationale': 'Compare alternatives.', 'evidence_ids': ['obs-1'], 'material_evidence_ids': []}


def fixture():
    contract = {'input_schema': {'type': 'object', 'properties': {'enabled': {'type': 'boolean'}},
        'required': ['enabled'], 'additionalProperties': False},
        'description': '开关仪器电源', 'source': 'generic native source',
        'preconditions': [{'description': 'Requires no running operation.', 'public_state_ids': ['status.busy']}],
        'execution': {'mode': 'synchronous', 'time_domain': 'simulation', 'cancellation': 'unsupported'},
        'limitations': ['Long global background text.' * 500],
        'output_schema': {'type': 'object', 'properties': {'accepted': {'type': 'boolean'}}},
        'observes': {'power.enabled': {'unit': '1', 'description': 'Power'}}}
    snapshot = {'id': 'obs-1', 'revision': 2, 'sample_id': 'instrument', 'quality': 'valid',
        'captured_at_step': 3, 'native': {'clock': {'time_s': 10, 'domain': 'simulation'},
        'observations': {'power.enabled': {'value': True, 'unit': '1', 'quality': 'estimated',
            'sample_time_s': 10, 'sample_window_s': 2},
        'reactor.indicated_k': {'value': 298.15, 'unit': 'K', 'quality': 'valid', 'sample_time_s': 10}}}}
    obs = {'inspection': deepcopy(snapshot), 'measurement': deepcopy(snapshot),
        'simulated_seconds': 20, 'revision': 3, 'last_mutation_step': 4,
        'pending_operation': {'operation_id': 'pending-1', 'status': 'running'},
        'capability_detail': {'capability_id': 'device::power.set', 'contract': deepcopy(contract),
            'simulation_time_s': 20, 'is_measurement': False},
        'last_action': {'proposal': {'capability_id': 'workflow.await', 'arguments': {'duration_s': 10}},
                        'error': {'code': 'OPERATION_PENDING'}}}
    return {'observations': obs, 'target': {'objective': 'Keep 24-26 degC', 'constraints': ['fixed batch']},
        'last_error': {'code': 'OPERATION_PENDING'}, 'hypotheses': [{'cause': 'Unverified hypothesis'}],
        'candidate_capabilities': {'device::power.set': contract, 'irrelevant': contract}}


def test_identical_snapshots_are_referenced_and_all_facts_preserved():
    context = fixture()
    before = deepcopy(context)
    out = compact_context(context, [candidate()])
    assert out['observations']['inspection'] == context['observations']['inspection']
    assert out['observations']['measurement'] == {'snapshot_ref': 'inspection'}
    for key in ['simulated_seconds', 'revision', 'last_mutation_step', 'pending_operation', 'last_action']:
        assert out['observations'][key] == context['observations'][key]
    for key in ['target', 'last_error', 'hypotheses']:
        assert out[key] == context[key]
    assert context == before


@pytest.mark.parametrize('change', ['time', 'value', 'quality', 'revision', 'sample_id', 'id'])
def test_snapshots_with_different_evidence_are_not_deduplicated(change):
    context = fixture()
    measurement = context['observations']['measurement']
    if change == 'time':
        measurement['native']['observations']['power.enabled']['sample_time_s'] = 20
    elif change == 'value':
        measurement['native']['observations']['power.enabled']['value'] = False
    elif change == 'quality':
        measurement['native']['observations']['power.enabled']['quality'] = 'unavailable'
    else:
        measurement[change] = 'different'
    out = compact_context(context, [candidate()])
    assert out['observations']['measurement'] == measurement
    assert out['observations']['inspection'] == context['observations']['inspection']


def test_contract_keeps_exact_schemas_units_and_prerequisites_without_manual():
    context = fixture()
    out = compact_context(context, [candidate()])
    assert set(out['candidate_capabilities']) == {'device::power.set'}
    contract = out['candidate_capabilities']['device::power.set']
    source = context['candidate_capabilities']['device::power.set']
    for key in ['input_schema', 'output_schema', 'preconditions', 'execution', 'observes', 'description']:
        assert contract[key] == source[key]
    assert 'limitations' not in contract
    detail = out['observations']['capability_detail']
    assert detail['contract'] == {'contract_ref': 'candidate_capabilities.device::power.set'}
    assert detail['is_measurement'] is False and detail['simulation_time_s'] == 20


def test_query_contract_stays_small_when_not_a_current_candidate():
    context = fixture()
    out = compact_context(context, [candidate('wait', 'workflow.await', {'duration_s': 2})])
    assert 'candidate_capabilities' not in out
    detail = out['observations']['capability_detail']['contract']
    assert 'limitations' not in detail
    assert detail['input_schema'] == context['observations']['capability_detail']['contract']['input_schema']


def test_candidates_appear_once_in_criteria_with_utf8_and_exact_arguments():
    context = fixture()
    options = [candidate(), candidate('read', 'workflow.readback', {'state_ids': ['power.enabled']})]
    before = deepcopy(options)
    body, projected = operation_request(context, options, MODEL)
    assert [json.loads(v) for v in body['questions']['next_action']['criteria'].values()] == options
    assert 'candidate_id' not in body['state']
    assert '\\u' not in body['state']
    assert '开关仪器电源' in body['state']
    assert json.loads(body['state'].split('Public context:\n')[1]) == projected
    assert options == before
    assert len(json.dumps(body, ensure_ascii=False).encode('utf-8')) < 8000


def test_empty_and_device_selection_contexts_remain_supported():
    for context in [{}, {'task': {'objective': 'Find instrument'}, 'described_devices': [{'id': 'x'}]}]:
        out = compact_context(context, [candidate('instrument', 'catalog.bind', {'device_id': 'x'})])
        assert out == context


def test_live_transport_sends_compact_request_and_preserves_native_choice(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    context, options, sent = fixture(), [candidate(), candidate('wait', 'workflow.await', {'duration_s': 2})], []
    async def post(self, url, **kwargs):
        body = kwargs['json']
        sent.append(body)
        assert 'Long global background' not in body['state']
        assert 'candidate_id' not in body['state']
        assert json.loads(body['questions']['next_action']['criteria']['A'])['intent']['arguments'] == {'enabled': False}
        return httpx.Response(200, json={'model': MODEL, 'answers': {'next_action': {
            'type': 'choice', 'choice': 'B', 'probabilities': {'A': .2, 'B': .9}}},
            'usage': {'input_tokens': 100, 'output_tokens': 2}})
    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    chooser = OpenRouterJevOperationBackend(TokenLedger(tmp_path/'ledger.jsonl'))
    decision = validate_decision(chooser.choose(context, options), options)
    assert decision.choice == 'wait' and decision.probabilities == {'power': .2, 'wait': .9}
    assert len(sent) == 1
    assert chooser.last_call['context'] == context
    assert chooser.last_call['candidates'] == options
    assert chooser.last_call['input_projection'] == POLICY_VERSION
    assert chooser.last_call['request_body_bytes'] == len(json.dumps(sent[0], ensure_ascii=False).encode('utf-8'))
    assert chooser.ledger.snapshot()['charged_tokens'] == 102


def test_projection_never_truncates_actual_state_to_fit_budget(tmp_path):
    context = fixture()
    context['observations']['inspection']['native']['observations']['large_result'] = {
        'value': 'Actual result ' * 4000, 'unit': '1', 'quality': 'valid'}
    body, projected = operation_request(context, [candidate()], MODEL)
    assert projected['observations']['inspection'] == context['observations']['inspection']
    with pytest.raises(ValueError, match='conservative debug limit'):
        TokenLedger(tmp_path/'ledger.jsonl').reserve_decision(body, provider='openrouter')


def test_autonomous_context_supplies_only_compared_public_contracts():
    from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
    from app.single_instrument import InstrumentTask
    task = InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8'))
    device = Rc1mxExperiment(task.model_copy(update={'device': 'mt_rc1mxhfcal'}))
    options = [candidate('power', 'mt_rc1mxhfcal::power.set', {'enabled': True})]
    context = {'capabilities': device.capabilities(), 'observations': device.observe()}
    before = deepcopy(device.lifecycle.calls)
    projected = device.operation_context(context, options)
    assert list(projected['candidate_capabilities']) == ['mt_rc1mxhfcal::power.set']
    assert projected['candidate_capabilities']['mt_rc1mxhfcal::power.set']['input_schema'] == context['capabilities']['mt_rc1mxhfcal::power.set']['input_schema']
    assert device.lifecycle.calls == before  # The compactor never reads the simulator.
