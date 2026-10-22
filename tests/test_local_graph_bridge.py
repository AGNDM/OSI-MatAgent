"""Offline integration checks of David's unmodified graph adapter."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark/graph'))
from graph_local.policy import GraphNativePolicy, GraphPolicyError
from app.models import OperationDecision, RouteDecision


class Menus:
    def action_options(self, view, history):
        if not view['selected_instrument']:
            return [{'kind': 'select_instrument', 'instrument_id': 'device'}]
        return [
            {'kind': 'invoke', 'instrument_id': 'device', 'capability_id': 'power.set', 'arguments': {'on': True}},
            {'kind': 'read', 'instrument_id': 'device', 'state_ids': ['powered']},
            {'kind': 'finish'},
        ]

    def argument_options(self, view, history, option):
        return [{'action': deepcopy(option)}]


class Expert:
    backend_id = 'offline-test-expert'
    is_llm = False

    def invoke(self, role, prompt, payload, schema):
        if role == 'diagnostic':
            view = payload['benchmark_public_context']['current_public_view']
            if not view['selected_instrument']:
                action = {'kind': 'select_instrument', 'instrument_id': 'device'}
            elif not view['observation']['powered']['value']:
                action = {'kind': 'invoke', 'instrument_id': 'device', 'capability_id': 'power.set', 'arguments': {'on': True}}
            else:
                action = {'kind': 'finish'}
            return {'hypotheses': [], 'candidates': [{'candidate_id': 'one', 'intent': {
                'capability_id': 'benchmark.' + action['kind'], 'arguments': {'action': action},
                'discriminating_observation': 'Native feedback', 'postcondition': 'Goal progress'},
                'rationale': 'Offline handoff test', 'evidence_ids': []}]}
        if role == 'procedure':
            intent = deepcopy(payload['intent']); intent.pop('discriminating_observation')
            return {**intent, **{k: payload[k] for k in ['sample_id', 'expected_revision', 'request_id']}}
        if role == 'verification':
            return {'status': 'supported', 'evidence_ids': [payload['observations']['measurement']['id']], 'missing_evidence': []}
        raise AssertionError(role)


class Decision:
    backend_id = 'offline-test-decision'
    is_llm = False
    is_decision_model = True

    def decide(self, summary, allowed):
        selected = 'verify' if allowed == ['verify'] else 'prepare_action' if summary['has_intent'] else 'diagnose'
        return RouteDecision(backend_id=self.backend_id, choice=selected,
                             probabilities={key: float(key == selected) for key in allowed})

    def choose(self, context, candidates):
        return OperationDecision(backend_id=self.backend_id, choice=candidates[0]['candidate_id'],
                                 probabilities={c['candidate_id']: float(i == 0) for i, c in enumerate(candidates)})


def view(selected=None, powered=False):
    return {'goal': 'Power on', 'selected_instrument': selected,
            'observation': {'powered': {'value': powered, 'unit': '1', 'quality': 'valid'}},
            'budget': {'max_decisions': 8}}


def policy(mode):
    return GraphNativePolicy({'task_id': 'offline', 'goal': 'Power on'}, Menus(), Expert(), Decision(), mode=mode)


def ack(proposal, executed=None, corrected=False):
    return {'proposed_action': proposal, 'executed_action': executed or proposal,
            'corrected': corrected, 'response': {'status': 'succeeded'}}


@pytest.mark.parametrize('mode', ['free_run', 'teacher_forced'])
def test_persistent_graph_hands_off_actions_and_receives_native_feedback(mode):
    p = policy(mode)
    first = p.next_action(view(), [])
    assert first == {'kind': 'select_instrument', 'instrument_id': 'device'}
    graph = p.graph
    history = [ack(first)]
    second = p.next_action(view('device'), history)
    assert second['kind'] == 'invoke' and second['arguments'] == {'on': True}
    assert p.lab.view['observation']['powered']['value'] is False
    history.append(ack(second))
    assert p.next_action(view('device', True), history) == {'kind': 'finish'}
    assert p.graph is graph
    assert p.lab.history == history


def test_teacher_corrected_finish_resumes_same_graph_with_real_feedback():
    p = policy('teacher_forced')
    first = p.next_action(view('device', True), [])
    assert first == {'kind': 'finish'}
    graph = p.graph
    correction = {'kind': 'read', 'instrument_id': 'device', 'state_ids': ['powered']}
    history = [ack(first, correction, True)]
    assert p.next_action(view('device', False), history)['kind'] == 'invoke'
    assert p.graph is graph and p.lab.history == history


def test_free_run_rejects_reference_correction():
    p = policy('free_run')
    proposal = p.next_action(view('device'), [])
    correction = {'kind': 'read', 'instrument_id': 'device', 'state_ids': ['powered']}
    with pytest.raises(GraphPolicyError, match='unauthorized native action replacement'):
        p.next_action(view('device'), [ack(proposal, correction, True)])


def test_mismatched_native_acknowledgement_is_rejected():
    p = policy('teacher_forced')
    p.next_action(view(), [])
    with pytest.raises(GraphPolicyError):
        p.next_action(view('device'), [ack({'kind': 'finish'})])
