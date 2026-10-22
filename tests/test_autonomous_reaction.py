"""Isolation, control freedom, and actual graph/selection integration checks."""
from copy import deepcopy
import json

import pytest

from app.autonomous_instrument import AutonomousInstrumentProfile, selection_options, DEVICE_CHOICES
from app.graph import build_graph, initial_state
from app.instrument_cli import run_episode
from app.lab_adapter import GateRejected
from app.models import ActionProposal, Limits, OperationDecision
from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
from app.single_instrument import InstrumentLab, InstrumentTask
from app.trace import Trace


def task():
    return InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8'))


def lab():
    return Rc1mxExperiment(task().model_copy(update={'device': 'mt_rc1mxhfcal'}))


def act(device, name, arguments=None):
    return ActionProposal(sample_id=device.sample_id, expected_revision=device.revision,
        capability_id=name, arguments=arguments or {}, request_id=f'{device.action_count}-{name}',
        postcondition='Offline behavior check')


def test_controls_do_not_require_or_create_observations():
    device = lab()
    for name, args in [('thermostat.set_tr', {'end_k': 298.15, 'ramp_s': 0, 'controller': 'aqueous'}),
                       ('stirrer.set', {'rpm': 250})]:
        event = device.execute(act(device, 'mt_rc1mxhfcal::' + name, args))
        assert not event['result'].get('error')
        assert device.observe()['inspection'] is None
        assert device.observe()['measurement'] is None
    assert device.action_count == 2
    assert device.lifecycle.calls.get('agent_read', 0) == 0
    assert device.lifecycle.calls['monitor_read'] > 0
    assert device.samples[-1]['thermostat_mode'] == 'tr'


def test_chosen_fields_and_historical_timestamp_survive_private_monitoring():
    device = lab()
    device.execute(act(device, 'workflow.readback', {'state_ids': ['power.enabled']}))
    old = deepcopy(device.observe()['inspection'])
    assert list(old['native']['observations']) == ['power.enabled']
    event = device.execute(act(device, 'workflow.await', {'duration_s': 2}))
    assert event['result']['advanced_seconds'] == 2
    assert event['result']['constraint_reads']  # Independent audit record.
    obs = device.observe()
    assert obs['measurement'] is None and obs['inspection'] == old
    assert 'constraint_reads' not in obs['last_action']['result']
    assert 'constraint_observation' not in obs['last_action']['result']
    assert old['native']['clock']['time_s'] == 0
    assert device.lifecycle.calls['agent_read'] == 1
    assert device.samples[-1]['time_s'] == 2
    device.execute(act(device, 'workflow.readback', {'state_ids': ['reactor.indicated_k']}))
    measured = device.observe()['measurement']['native']
    assert list(measured['observations']) == ['reactor.indicated_k']
    assert measured['clock']['time_s'] == 2
    assert device.lifecycle.calls['agent_read'] == 2


def test_private_field_read_is_a_boundary():
    device = lab()
    with pytest.raises(GateRejected):
        device.execute(act(device, 'workflow.readback', {'state_ids': ['truth.composition_mol']}))
    assert device.termination_reason == 'capability_boundary'
    assert device.action_count == 0


def test_wait_never_automatically_polls(monkeypatch):
    device = lab()
    device.pending_operation = {'operation_id': 'offline-pending', 'status': 'running', 'capability': 'calibration.run'}
    def forbidden(*args):
        raise AssertionError('Implicit poll')
    monkeypatch.setattr(device.lifecycle, 'poll', forbidden)
    result = device.execute(act(device, 'workflow.await', {'duration_s': 2}))
    assert not result['result'].get('error')
    assert result['result']['polls'] == []
    assert device.pending_operation['operation_id'] == 'offline-pending'


class FirstOption:
    backend_id = 'offline-first-option'
    is_llm = False
    def choose(self, context, candidates):
        return OperationDecision(backend_id=self.backend_id, choice=candidates[0]['candidate_id'],
            probabilities={c['candidate_id']: 1 if i == 0 else 0 for i, c in enumerate(candidates)})


class OfflineRoles:
    backend_id = 'offline-autonomy-test'
    is_llm = False
    def __init__(self, selected='mt_rc1mxhfcal'):
        self.selected = selected
        self.diagnostic_inputs = []
        self.diagnostic_prompts = []
    def invoke(self, role, prompt, payload, schema):
        if role == 'device_discovery':
            assert payload['task'] == {'objective': task().objective}
            assert [d['id'] for d in payload['device_directory']] == list(DEVICE_CHOICES)
            assert 'mt_rc1' not in json.dumps(payload['task']).lower()
            if not payload['discovered_contracts']:
                return {'action': 'describe', 'device_ids': [self.selected]}
            contract = payload['discovered_contracts'][self.selected]
            assert 'capabilities' not in contract and 'capability_ids' in contract
            return {'action': 'propose', 'candidates': [dict(candidate_id='selected-device',
                device_id=self.selected, rationale='Offline selection forwarding check')]}
        if role == 'diagnostic':
            self.diagnostic_inputs.append(deepcopy(payload))
            self.diagnostic_prompts.append(prompt)
            assert 'propose inspection only' not in prompt
            assert 'return one workflow.inspect' not in prompt
            assert payload['observations']['inspection'] is None
            assert payload['observations']['measurement'] is None
            count = len(self.diagnostic_inputs)
            if count == 1:
                cap, args = 'mt_rc1mxhfcal::stirrer.set', {'rpm': 250}
            elif count == 2:
                cap, args = 'workflow.await', {'duration_s': 2}
            else:
                cap, args = 'workflow.stop', {'reason': 'Offline integration test complete'}
            return {'hypotheses': [], 'candidates': [dict(candidate_id=f'action-{count}',
                intent=dict(capability_id=cap, arguments=args, discriminating_observation='Test',
                    postcondition='Test'), rationale='Offline test', evidence_ids=[])]}
        if role == 'procedure':
            return {k: payload[k] for k in ('sample_id', 'expected_revision', 'request_id')} | {
                k: payload['intent'][k] for k in ('capability_id', 'arguments', 'postcondition')}
        raise AssertionError(role)


def test_full_auto_selection_forwards_factory_and_keeps_monitor_data_private(tmp_path):
    roles = OfflineRoles()
    devices = []
    def factory(selected):
        device = Rc1mxExperiment(selected)
        devices.append(device)
        return device
    result = run_episode(task(), tmp_path/'run', roles, operation=FirstOption(),
        profile=AutonomousInstrumentProfile(), lab_factory=factory, router_wait=False,
        selection_options=selection_options(task()),
        limits=Limits(lab_actions=10, llm_calls=20, episode_seconds=30, graph_steps=100))
    assert result['selection']['selected_device'] == 'mt_rc1mxhfcal'
    assert result['selection']['catalog_count'] == 4
    assert len(devices) == 1
    assert result['report']['terminal_reason'] == 'agent_stopped_unresolved', result
    assert len(roles.diagnostic_inputs) == 3
    assert devices[0].lifecycle.elapsed == 2
    assert devices[0].lifecycle.calls.get('agent_read', 0) == 0
    for payload in roles.diagnostic_inputs:
        text = json.dumps(payload)
        assert 'truth.composition_mol' not in text and 'constraint_reads"' not in text
        assert 'predicates' not in payload['target']
        assert 'thermostat.mode' not in payload['target']
        assert 'Tr control with setpoint' not in text


def test_wrong_device_choice_is_not_silently_replaced(tmp_path):
    result = run_episode(task(), tmp_path/'wrong-device', OfflineRoles('ta_dsc2500'),
        operation=FirstOption(), profile=AutonomousInstrumentProfile(),
        lab_factory=Rc1mxExperiment, router_wait=False, selection_options=selection_options(task()),
        limits=Limits(lab_actions=10, llm_calls=20, episode_seconds=30, graph_steps=100))
    assert result['selection']['selected_device'] == 'ta_dsc2500'
    assert result['report']['terminal_reason'] == 'scenario_boundary'
    assert not result['score']['task_success']


def test_default_runner_still_requires_inspection():
    device = InstrumentLab(task().model_copy(update={'device': 'mt_rc1mxhfcal'}))
    with pytest.raises(GateRejected, match='INSPECTION_REQUIRED'):
        device.validate(act(device, 'mt_rc1mxhfcal::power.set', {'enabled': True}))
