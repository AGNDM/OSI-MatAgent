"""Regression: successful start survives waits without fabricating a fresh read."""
from copy import deepcopy
import json

import httpx
import pytest

from app.action_memory import attach_public_memory, operation_memory, snapshot_status
from app.autonomous_instrument import AutonomousInstrumentProfile
from app.decision_backend import RulesBackend
from app.instrument_cli import run_episode
from app.models import ActionProposal, Limits, OperationDecision
from app.openrouter_jev import OpenRouterJevBackend, OpenRouterJevOperationBackend, MODEL
from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
from app.single_instrument import InstrumentTask
from app.token_budget import TokenLedger


def task():
    return InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8')).model_copy(
        update={'device': 'mt_rc1mxhfcal'})


def act(device, name, args):
    return device.execute(ActionProposal(sample_id=device.sample_id, expected_revision=device.revision,
        capability_id=name, arguments=args, request_id=f'test-{device.action_count}', postcondition='Test'))


def receipts(obs):
    memory = obs['operation_memory']
    return [row for row in memory['recent_actions'] + memory['earlier_latest_receipts'] if 'receipt' in row]


def assert_started_with_old_snapshot(obs):
    started = [r for r in receipts(obs) if r['capability_id'] == 'mt_rc1mxhfcal::experiment.start']
    assert len(started) == 1
    assert started[0]['receipt']['status'] == 'succeeded'
    assert started[0]['receipt']['result']['experiment_id'] == 'exp-1'
    assert obs['inspection']['native']['observations']['experiment.running']['value'] is False
    assert obs['snapshot_status']['inspection']['status'] == 'stale'
    assert obs['measurement'] is None
    serialized = json.dumps(obs)
    for private in ('truth.composition_mol', 'constraint_reads"', 'constraint_observation"', 'conversion"'):
        assert private not in serialized


def test_native_start_and_waits_keep_success_without_hidden_reads(monkeypatch):
    device = Rc1mxExperiment(task())
    act(device, 'workflow.readback', {'state_ids': ['experiment.running', 'experiment.id']})
    old = deepcopy(device.observe()['inspection'])
    assert device.observe()['snapshot_status']['inspection']['status'] == 'current'
    started = act(device, 'mt_rc1mxhfcal::experiment.start', {'name': 'offline-memory'})
    obs = device.observe()
    assert_started_with_old_snapshot(obs)
    assert obs['snapshot_status']['inspection']['age_s'] == 0  # Same clock, changed revision.
    for _ in range(8):
        act(device, 'workflow.await', {'duration_s': 2})
    before_calls, before_history = deepcopy(device.lifecycle.calls), deepcopy(device.history)
    def forbidden(*args, **kwargs):
        raise AssertionError('Memory projection must not query the runtime')
    monkeypatch.setattr(device.session, 'read', forbidden)
    monkeypatch.setattr(device.lifecycle, 'poll', forbidden)
    obs = device.observe()
    assert_started_with_old_snapshot(obs)
    assert obs['inspection'] == old
    assert obs['snapshot_status']['inspection']['age_s'] == 16
    assert len(obs['operation_memory']['recent_actions']) == 6
    earlier = obs['operation_memory']['earlier_latest_receipts']
    assert len(earlier) == 1 and earlier[0]['step'] == started['step']
    assert device.lifecycle.calls == before_calls and device.history == before_history
    assert device.lifecycle.calls['agent_read'] == 1


def test_latest_receipts_keep_actual_outcome_and_do_not_rewrite_old_values():
    device = Rc1mxExperiment(task())
    act(device, 'mt_rc1mxhfcal::stirrer.set', {'rpm': 200})
    act(device, 'mt_rc1mxhfcal::stirrer.set', {'rpm': 300})
    for _ in range(7):
        act(device, 'workflow.await', {'duration_s': 2})
    memory = device.observe()['operation_memory']
    assert len(memory['earlier_latest_receipts']) == 1
    assert memory['earlier_latest_receipts'][0]['receipt']['result']['setpoint_rpm'] == 300
    assert all(r['time_s'] <= device.lifecycle.elapsed for r in memory['recent_actions'])
    act(device, 'mt_rc1mxhfcal::experiment.start', {'name': 'test'})
    rejected = act(device, 'mt_rc1mxhfcal::experiment.start', {'name': 'duplicate'})
    row = device.observe()['operation_memory']['recent_actions'][-1]
    assert row['receipt']['status'] == 'rejected'
    assert row['receipt']['error'] == rejected['result']['native_response']['error']
    assert row['receipt']['error']['code'] == 'EXPERIMENT_RUNNING'


def test_wait_memory_reports_actual_shortened_duration_without_monitor_data():
    event = {'step': 1, 'proposal': {'capability_id': 'workflow.await', 'arguments': {'duration_s': 120}},
        'result': {'advanced_seconds': 4, 'constraint_reads': [{'truth.composition_mol': 'private'}]}}
    before = deepcopy(event)
    row = operation_memory([event], 4)['recent_actions'][0]
    assert row == dict(step=1, time_s=4, capability_id='workflow.await', arguments={'duration_s': 120}, advanced_seconds=4)
    assert event == before


def test_missing_and_unknown_snapshots_are_not_marked_current():
    assert snapshot_status({})['inspection']['status'] == 'missing'
    status = snapshot_status({'simulated_seconds': 2, 'revision': 1,
        'inspection': {'id': 'incomplete', 'revision': 1, 'captured_at_step': 5}})
    assert status['inspection']['status'] == 'unknown'


class RememberingRoles:
    backend_id, is_llm = 'offline-memory-regression', False
    def __init__(self):
        self.inputs = []
        self.steps = [('workflow.readback', {'state_ids': ['experiment.running', 'experiment.id']}),
            ('mt_rc1mxhfcal::experiment.start', {'name': 'graph-memory'})] + [
            ('workflow.await', {'duration_s': 2})] * 8 + [('workflow.stop', {'reason': 'Offline test complete'})]
    def invoke(self, role, prompt, payload, schema):
        self.inputs.append((role, deepcopy(payload)))
        if role == 'diagnostic':
            index = sum(r == 'diagnostic' for r, _ in self.inputs) - 1
            cap, args = self.steps[index]
            return {'hypotheses': [], 'candidates': [{'candidate_id': f'option-{index}',
                'intent': {'capability_id': cap, 'arguments': args,
                    'discriminating_observation': 'Offline check', 'postcondition': 'Offline check'},
                'rationale': 'Offline check', 'evidence_ids': []}]}
        assert role == 'procedure'
        return {k: payload[k] for k in ('sample_id', 'expected_revision', 'request_id')} | {
            k: payload['intent'][k] for k in ('capability_id', 'arguments', 'postcondition')}


def test_full_graph_sends_memory_to_deepseek_roles_and_both_jev_transports(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-key')
    requests = []
    async def post(self, url, **kwargs):
        body = kwargs['json']
        requests.append(deepcopy(body))
        criteria = body['questions']['next_action']['criteria']
        if 'Public context:\n' in body['state']:
            choice = next(iter(criteria))  # Only one operation candidate in this test.
        else:
            summary = json.loads(body['state'].split('State:\n')[1].split('\n\nQuestion:')[0])
            route = 'prepare_action' if summary['has_intent'] else 'diagnose'
            choice = next(k for k, v in criteria.items() if v.startswith(route + ':'))
        return httpx.Response(200, json={'model': MODEL, 'answers': {'next_action': {
            'type': 'choice', 'choice': choice, 'probabilities': {k: int(k == choice) for k in criteria}}},
            'usage': {'input_tokens': 100, 'output_tokens': 1}})
    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    roles, devices = RememberingRoles(), []
    def factory(selected):
        device = Rc1mxExperiment(selected)
        devices.append(device)
        return device
    ledger = TokenLedger(tmp_path/'ledger.jsonl')
    result = run_episode(task(), tmp_path/'run', roles, router=OpenRouterJevBackend(ledger),
        operation=OpenRouterJevOperationBackend(ledger), profile=AutonomousInstrumentProfile(),
        lab_factory=factory, router_wait=False,
        limits=Limits(lab_actions=15, llm_calls=100, episode_seconds=60, graph_steps=200))
    assert result['report']['terminal_reason'] == 'agent_stopped_unresolved', result
    assert devices[0].lifecycle.calls['agent_read'] == 1
    last_diag = next(p for r, p in reversed(roles.inputs) if r == 'diagnostic')
    last_proc = next(p for r, p in reversed(roles.inputs) if r == 'procedure')
    assert_started_with_old_snapshot(last_diag['observations'])
    assert last_proc['operation_memory'] == last_diag['observations']['operation_memory']
    assert last_proc['snapshot_status'] == last_diag['observations']['snapshot_status']
    chooser_inputs = [json.loads(b['state'].split('Public context:\n')[1]) for b in requests
                      if 'Public context:\n' in b['state']]
    assert_started_with_old_snapshot(chooser_inputs[-1]['observations'])
    assert chooser_inputs[-1]['observations']['operation_memory'] == last_proc['operation_memory']
    router_inputs = [json.loads(b['state'].split('State:\n')[1].split('\n\nQuestion:')[0]) for b in requests
                     if 'Public context:\n' not in b['state']]
    assert router_inputs[-1]['operation_memory'] == last_proc['operation_memory']
    assert router_inputs[-1]['snapshot_status'] == last_proc['snapshot_status']
    assert max(len(json.dumps(b, ensure_ascii=False).encode()) for b in requests) < 24000
