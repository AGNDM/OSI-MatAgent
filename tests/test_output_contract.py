"""Check the model-facing schema, candidate validation, and native rejection separately."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.autonomous_instrument import AutonomousInstrumentProfile
from app.device_roles import DeviceProfile
from app.expert_config import CompatibleExpert, ExpertEndpoint
from app.graph import build_graph, initial_state
from app.models import ActionProposal, HypothesisSet, Intent, Limits, OperationOptions
from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
from app.single_instrument import InstrumentTask
from app.trace import Trace


def lab():
    task = InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8'))
    return Rc1mxExperiment(task.model_copy(update={'device': 'mt_rc1mxhfcal'}))


def options(capability='mt_rc1mxhfcal::power.set', arguments=None):
    return {'hypotheses': [], 'candidates': [{'candidate_id': 'candidate',
        'intent': {'capability_id': capability, 'arguments': {} if arguments is None else arguments,
                   'discriminating_observation': 'Offline test', 'postcondition': 'Offline test'},
        'rationale': 'Offline test', 'evidence_ids': []}]}


def test_serialized_model_schema_requires_arguments_without_a_default():
    for schema in (OperationOptions, HypothesisSet):
        definition = schema.model_json_schema()['$defs']['Intent']
        assert 'arguments' in definition['required']
        assert 'default' not in definition['properties']['arguments']
        assert 'input_schema' in definition['properties']['arguments']['description']
    assert 'arguments' in ActionProposal.model_json_schema()['required']


@pytest.mark.parametrize('bad', ['missing', None, 'enabled=true', []])
def test_missing_null_and_nonobject_arguments_fail_output_validation(bad):
    raw = options()
    if bad == 'missing':
        del raw['candidates'][0]['intent']['arguments']
    else:
        raw['candidates'][0]['intent']['arguments'] = bad
    before = deepcopy(raw)
    with pytest.raises(ValidationError) as error:
        OperationOptions.model_validate(raw)
    assert error.value.errors()[0]['loc'] == ('candidates', 0, 'intent', 'arguments')
    assert raw == before


@pytest.mark.parametrize('capability,args,accepted', [
    ('workflow.inspect', {}, True),
    ('mt_rc1mxhfcal::power.set', {}, False),
    ('mt_rc1mxhfcal::power.set', {'enabled': True}, True),
    ('mt_rc1mxhfcal::power.set', {'enabled': False}, True),
    ('mt_rc1mxhfcal::power.set', {'enabled': 'true'}, False),
    ('mt_rc1mxhfcal::power.set', {'enabled': True, 'extra': 1}, False),
    ('workflow.await', {}, False),
    ('workflow.await', {'duration_s': 2}, True),
    ('workflow.await', {'duration_s': 121}, False),
])
def test_selected_native_or_wrapper_schema_controls_parameters(capability, args, accepted):
    device = lab()
    record = OperationOptions.model_validate(options(capability, args))
    item = HypothesisSet(hypotheses=[], intent=record.candidates[0].intent)
    payload = {'capabilities': device.capabilities()}
    if accepted:
        DeviceProfile.validate_role_contract('diagnostic', item, payload)
        assert item.intent.arguments == args
    else:
        with pytest.raises(ValueError, match='WORKFLOW_ARGUMENT_VALIDATION: ' + capability):
            DeviceProfile.validate_role_contract('diagnostic', item, payload)
    assert device.lifecycle.calls['invoke'] == 0
    assert device.lifecycle.elapsed == 0


@pytest.mark.parametrize('missing', [False, True])
def test_openrouter_adapter_sends_required_arguments_schema_without_network(monkeypatch, missing):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-key')
    seen = []
    raw = options(arguments={'enabled': False})
    if missing:
        del raw['candidates'][0]['intent']['arguments']
    async def respond(self, body, timeout):
        seen.append(body)
        schema = json.loads(body['messages'][0]['content'].split('No Markdown or extra fields.\n', 1)[1])
        assert 'arguments' in schema['$defs']['Intent']['required']
        assert 'Every candidate must be fully parameterized' in body['messages'][0]['content']
        return {'choices': [{'message': {'content': json.dumps(raw)}, 'finish_reason': 'stop'}]}
    monkeypatch.setattr(CompatibleExpert, '_request', respond)
    backend = CompatibleExpert(ExpertEndpoint(name='offline-contract', provider='openrouter',
        base_url='https://openrouter.ai/api/v1', model='deepseek/deepseek-v4.1-flash',
        api_key_env='OPENROUTER_API_KEY'))
    profile = AutonomousInstrumentProfile()
    if missing:
        with pytest.raises(ValidationError, match='arguments'):
            backend.invoke('diagnostic', profile.candidate_prompt + profile.candidate_instruction, {}, OperationOptions)
    else:
        parsed = backend.invoke('diagnostic', profile.candidate_prompt + profile.candidate_instruction, {}, OperationOptions)
        assert parsed['candidates'][0]['intent']['arguments'] == {'enabled': False}
    assert len(seen) == 1


def test_graph_stops_invalid_candidate_before_chooser_gate_or_native_call():
    device = lab()
    class Roles:
        backend_id = 'offline-invalid-candidate'
        is_llm = False
        def invoke(self, role, prompt, payload, schema):
            assert role == 'diagnostic'
            return options(arguments={})
    class Chooser:
        backend_id = 'must-not-be-called'
        is_llm = False
        def choose(self, *args):
            pytest.fail('Invalid candidates must not reach operation selection')
    limits = Limits(lab_actions=3, graph_steps=30, format_repairs=0)
    trace = Trace('invalid-arguments-layer')
    graph = build_graph(device, trace, limits, roles=Roles(), operation_backend=Chooser(),
                        profile=AutonomousInstrumentProfile())
    end = graph.invoke(initial_state(device, trace.episode_id, limits),
                       {'configurable': {'thread_id': trace.episode_id}, 'recursion_limit': 30})
    assert end['report']['terminal_reason'] == 'role_error'
    failure = next(e for e in trace.events if e['kind'] == 'role_error')
    assert 'WORKFLOW_ARGUMENT_VALIDATION' in failure['error']
    assert not any(e['component'] == 'gate' or e['kind'] == 'tool_result' for e in trace.events)
    assert device.lifecycle.calls['invoke'] == 0 and device.action_count == 0


def test_native_runtime_also_rejects_missing_arguments_independently():
    device = lab()
    before = device.session.read(instance_id=device.instance_id, state_ids=['power.enabled'])
    # Bypass the workflow in this isolated offline test to check the native boundary itself.
    result = device.session.invoke(instance_id=device.instance_id, capability_id='power.set',
                                   arguments={}, request_id='offline-missing-enabled')
    after = device.session.read(instance_id=device.instance_id, state_ids=['power.enabled'])
    assert result['status'] == 'rejected'
    assert result['error']['code'] == 'INVALID_ARGUMENT'
    assert before == after
    assert device.lifecycle.elapsed == 0
