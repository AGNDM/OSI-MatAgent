"""A stale observation must not hide a newly selected readback from the router."""
from copy import deepcopy
import json

import httpx
import pytest

from app.autonomous_instrument import AutonomousInstrumentProfile
from app.decision_backend import summarize
from app.graph import initial_state
from app.instrument_cli import run_episode
from app.local_decision import option_prompt
from app.models import Limits
from app.openrouter_jev import OpenRouterJevBackend, OpenRouterJevOperationBackend, MODEL
from app.rc1mx_experiment import Rc1mxExperiment
from app.router_context import selected_intent_context
from app.token_budget import TokenLedger
from tests.test_action_memory import act, task, RememberingRoles


def selected_state(device):
    state = initial_state(device, 'router-regression', Limits())
    state['observations'] = device.observe()
    state['intent'] = dict(capability_id='workflow.readback',
        arguments={'state_ids':['reactor.indicated_k', 'clock.time_s']},
        discriminating_observation='Refresh the previous observation', postcondition='Read chosen fields')
    state['operation_decision'] = {'choice':'read-current-temperature'}
    state['operation_revision'] = device.revision
    return state


def test_current_readback_remains_current_with_old_sensor_snapshot_and_no_io(monkeypatch):
    device = Rc1mxExperiment(task())
    act(device, 'workflow.readback', {'state_ids':['reactor.indicated_k']})
    act(device, 'workflow.await', {'duration_s':2})
    state = selected_state(device)
    before = deepcopy(state)
    calls = deepcopy(device.lifecycle.calls)
    def forbidden(*args, **kwargs):
        raise AssertionError('Router projection must not read or poll')
    monkeypatch.setattr(device.session, 'read', forbidden)
    monkeypatch.setattr(device.lifecycle, 'poll', forbidden)
    allowed = ['diagnose','prepare_action']
    summary, returned = device.route_context(state, summarize(state), allowed)
    assert summary['selected_intent'] == state['intent']
    assert summary['selected_intent_revision'] == summary['current_revision'] == 1
    assert summary['selected_intent_current'] is True
    assert summary['snapshot_status']['inspection']['read_revision'] == 0
    assert summary['snapshot_status']['inspection']['status'] == 'stale'
    text, options = option_prompt(summary, returned)
    transmitted = json.loads(text.split('State:\n')[1].split('\n\nQuestion:')[0])
    assert transmitted == summary
    assert 'prepare it instead of planning the same read again' in text
    assert 'Use when no actionable intent or fresh measurement exists' not in text
    assert returned == allowed and set(options.values()) == set(allowed)
    assert state == before and device.lifecycle.calls == calls


@pytest.mark.parametrize('change,expected_revision', [
    ({'operation_revision': -1}, -1),
    ({'observation_revision': -1}, 0),
    ({'operation_decision': None}, None),
    ({'intent': None}, None),
])
def test_stale_or_unconfirmed_selection_is_never_labelled_current(change, expected_revision):
    device = Rc1mxExperiment(task())
    state = selected_state(device)
    state.update(change)
    summary = selected_intent_context(state, device.revision)
    assert summary['selected_intent_current'] is False
    assert summary['selected_intent_revision'] == expected_revision


def test_new_error_is_forwarded_without_overriding_router_choice():
    device = Rc1mxExperiment(task())
    state = selected_state(device)
    state['last_error'] = {'code':'NEW_ERROR'}
    summary, allowed = device.route_context(state, summarize(state), ['diagnose','prepare_action'])
    assert summary['last_error'] == state['last_error']
    assert summary['selected_intent_current'] is True  # Revision freshness and errors are separate.
    assert allowed == ['diagnose','prepare_action']
    text, _ = option_prompt(summary, allowed)
    assert 'a new error requires replanning' in text


def test_real_graph_transmits_selected_read_and_revision_to_jev_http(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-key')
    routed = []
    async def post(self, url, **kwargs):
        body = kwargs['json']
        criteria = body['questions']['next_action']['criteria']
        if 'Public context:\n' in body['state']:
            choice = next(iter(criteria))
        else:
            summary = json.loads(body['state'].split('State:\n')[1].split('\n\nQuestion:')[0])
            routed.append((summary, deepcopy(body)))
            route = 'prepare_action' if summary['selected_intent_current'] and not summary['last_error'] else 'diagnose'
            choice = next(k for k,v in criteria.items() if v.startswith(route + ':'))
        return httpx.Response(200, json={'model':MODEL, 'answers':{'next_action':{
            'type':'choice','choice':choice,'probabilities':{k:int(k==choice) for k in criteria}}},
            'usage':{'input_tokens':100,'output_tokens':1}})
    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    roles, devices = RememberingRoles(), []
    roles.steps = [('workflow.readback',{'state_ids':['reactor.indicated_k']}),
        ('workflow.await',{'duration_s':2}),
        ('workflow.readback',{'state_ids':['reactor.indicated_k','clock.time_s']}),
        ('workflow.stop',{'reason':'Offline test complete'})]
    def factory(selected):
        device = Rc1mxExperiment(selected)
        devices.append(device)
        return device
    ledger = TokenLedger(tmp_path/'ledger.jsonl')
    result = run_episode(task(),tmp_path/'run',roles, router=OpenRouterJevBackend(ledger),
        operation=OpenRouterJevOperationBackend(ledger),profile=AutonomousInstrumentProfile(),
        lab_factory=factory,router_wait=False,limits=Limits(lab_actions=8,llm_calls=80,episode_seconds=60,graph_steps=100))
    assert result['report']['terminal_reason'] == 'agent_stopped_unresolved', result
    summaries = [s for s,b in routed]
    chosen = [s for s in summaries if s['selected_intent_current']]
    assert len(chosen) == 4
    assert [(s['selected_intent']['capability_id'],s['selected_intent']['arguments']) for s in chosen] == roles.steps
    after_wait = chosen[2]
    assert after_wait['selected_intent_revision'] == after_wait['current_revision'] == 1
    assert after_wait['snapshot_status']['inspection']['read_revision'] == 0
    assert after_wait['snapshot_status']['inspection']['status'] == 'stale'
    assert devices[0].lifecycle.calls['agent_read'] == 2
    assert devices[0].observe()['inspection']['native']['clock']['time_s'] == 2
    assert all(s['selected_intent'] is None for s in summaries if not s['has_intent'])
    assert max(len(json.dumps(b,ensure_ascii=False).encode()) for s,b in routed) < 24000
