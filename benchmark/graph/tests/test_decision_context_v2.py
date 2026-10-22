"""Native state/criteria placement checks; run only in the Slurm test job."""
from copy import deepcopy
import json

import pytest

from app.models import ActionCandidate
from graph_local.backends import LocalDecisionBackend
from graph_local.policy import PublicBenchmarkLab


class CapturingChooser:
    identity = {'backend': 'test-double-not-a-model'}

    def __init__(self):
        self.calls = []

    def decide(self, messages, labels, **kwargs):
        self.calls.append({'messages': deepcopy(messages), 'labels': list(labels),
                           'kwargs': deepcopy(kwargs)})
        return {'status': 'ok', 'selected_option_id': labels[-1],
                'probabilities': {label: float(label == labels[-1]) for label in labels},
                'input_tokens': 50, 'output_tokens': 0}


def context():
    return {'observations': {'measurement': {'id': 'native-reading-7', 'quality': 'valid'}},
        'last_error': {'code': 'BUSY'},
        'benchmark_public_context': {'current_public_view': {
            'goal': 'Observe completed native program', 'selected_instrument': 'furnace',
            'observation': {'observations': {'temperature': {'value': 313.15, 'unit': 'K',
                'quality': 'estimated', 'sample_time_s': 25}}},
            'manual': ['An accepted command does not prove completion.'],
            'budget': {'decisions_used': 7, 'max_decisions': 32}, 'elapsed_sim_time_s': 25,
            'contract': {}}, 'complete_native_history': [], 'wrapper_revision': 7}}


def candidates():
    return [{'candidate_id': name,
        'intent': {'capability_id': 'benchmark.' + action['kind'], 'arguments': {'action': action},
            'discriminating_observation': 'Use exact current reading native-reading-7.',
            'postcondition': 'Preserve the supplied native semantics.'},
        'rationale': 'EXACT_CANDIDATE_RATIONALE_' + name, 'evidence_ids': ['native-reading-7']}
        for name, action in (
            ('read-current', {'kind': 'read', 'instrument_id': 'furnace', 'state_ids': ['program.phase']}),
            ('wait-offered', {'kind': 'wait', 'dt_s': 30}))]


@pytest.mark.parametrize('family', ['openjev', 'apus'])
def test_v2_native_candidates_live_once_in_criteria_with_full_state_and_audit(family):
    fake = CapturingChooser()
    chooser = LocalDecisionBackend({'kind': 'native', 'family': family,
        'context_protocol': 'decision-view-v2'}, backend=fake)
    source, offered = context(), candidates()
    original = deepcopy((source, offered))
    decision = chooser.choose(source, offered)
    call = fake.calls[0]
    state = call['messages'][0]['content']
    assert 'Candidate operations:' not in state
    assert 'EXACT_CANDIDATE_RATIONALE_' not in state
    view = json.loads(state.split('\nPublic context:\n', 1)[1])
    assert view['protocol'] == 'decision-view-v2'
    assert view['state']['current_observation'] == source['benchmark_public_context']['current_public_view']['observation']
    assert view['state']['last_error'] == {'code': 'BUSY'}
    assert view['audit']['input_sha256']
    assert 'candidates' not in view
    normalized = [ActionCandidate.model_validate(value).model_dump(mode='json') for value in offered]
    actual = [json.loads(value) for value in call['kwargs']['option_descriptions'].values()]
    assert actual == normalized
    assert decision.choice == 'wait-offered'
    assert chooser.last_call['messages'] == call['messages']
    assert chooser.last_call['public_input']['context'] == source
    assert chooser.last_call['public_input']['candidates'] == normalized
    assert chooser.last_call['public_input']['decision_view'] == view
    assert chooser.manifest()['native_option_placement'] == 'question_criteria_only'
    assert (source, offered) == original


@pytest.mark.parametrize('kind,protocol', [('hf', 'decision-view-v2'),
    ('native', 'decision-view-v1'), ('native', None)])
def test_hf_v2_and_earlier_native_protocols_keep_their_full_option_list(kind, protocol):
    fake = CapturingChooser()
    specification = {'kind': kind}
    if protocol is not None:
        specification['context_protocol'] = protocol
    chooser = LocalDecisionBackend(specification, backend=fake)
    chooser.choose(context(), candidates())
    text = fake.calls[0]['messages'][0]['content']
    assert '\nCandidate operations:\n' in text
    for value in candidates():
        assert value['candidate_id'] in text
        assert value['rationale'] in text
    assert 'native_option_placement' not in chooser.manifest()


def test_v2_native_router_keeps_route_ids_and_exact_criteria_in_separate_channels():
    fake = CapturingChooser()
    chooser = LocalDecisionBackend({'kind': 'native', 'context_protocol': 'decision-view-v2'}, backend=fake)
    summary = context()
    summary.update(has_intent=True, intent=candidates()[0]['intent'])
    chooser.decide(summary, ['diagnose', 'prepare_action'])
    call = fake.calls[0]
    state = call['messages'][0]['content']
    assert '\nOptions:\n' not in state
    view = json.loads(state.split('\nState:\n', 1)[1])
    assert view['protocol'] == 'decision-view-v2'
    assert view['allowed_routes'] == ['diagnose', 'prepare_action']
    assert view['state']['selected_intent'] == summary['intent']
    criteria = call['kwargs']['option_descriptions']
    assert set(criteria) == {'A', 'B'}
    assert criteria['A'].startswith('diagnose: ')
    assert criteria['B'].startswith('prepare_action: ')
    assert all(description not in state for description in criteria.values())


@pytest.mark.parametrize('protocol', ['decision-view-v1', 'decision-view-v2'])
def test_policy_forwards_current_role_state_for_both_declared_decision_views(protocol):
    lab = object.__new__(PublicBenchmarkLab)
    lab.decision_context_protocol = protocol
    public = context()['benchmark_public_context']
    lab.context = lambda: deepcopy(public)
    state = {key: {'marker': key} for key in ('observations', 'target', 'intent', 'hypotheses',
        'last_error', 'verification', 'pending_operation', 'action_count', 'llm_call_count')}
    summary = {'has_intent': True}
    result, routes = lab.route_context(state, summary, ['diagnose', 'prepare_action'])
    assert all(result[key] == value for key, value in state.items())
    assert result['benchmark_public_context'] == public
    assert routes == ['diagnose', 'prepare_action']
    assert summary == {'has_intent': True}
