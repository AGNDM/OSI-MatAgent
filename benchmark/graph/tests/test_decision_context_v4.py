"""V4 preserves native evidence/criteria while omitting parameter-schema copies."""
from copy import deepcopy
import json

import pytest

from app.models import ActionCandidate
from graph_local.backends import LocalDecisionBackend
from test_decision_context_v2 import CapturingChooser, candidates, context


@pytest.mark.parametrize('kind', ['native', 'hf'])
def test_v4_omits_only_model_parameter_schemas_and_preserves_exact_criteria(kind):
    source = context()
    offered = candidates()
    offered[0]['intent']['arguments']['action'] = {'kind': 'invoke', 'instrument_id': 'furnace',
        'capability_id': 'power.set', 'arguments': {'enabled': False}}
    source['benchmark_public_context']['current_public_view']['contract'] = {'capabilities': {
        'power.set': {'input_schema': {'type': 'object', 'properties': {'enabled': {'type': 'boolean'}}},
                      'description': 'Set power', 'preconditions': ['Public precondition'],
                      'output_schema': {'type': 'object'}}}}
    before = deepcopy(source)
    fake = CapturingChooser()
    chooser = LocalDecisionBackend({'kind': kind, 'context_protocol': 'decision-view-v4'}, backend=fake)
    chooser.choose(source, offered)
    full = chooser.last_call['public_input']['decision_view']
    model = chooser.last_call['public_input']['model_decision_view']
    assert model['state'] == full['state']
    assert model['state']['current_observation'] == source['benchmark_public_context']['current_public_view']['observation']
    for name, original in full['candidate_contracts']['capabilities'].items():
        expected = deepcopy(original)
        expected.pop('input_schema', None)
        expected.pop('output_schema', None)
        assert model['candidate_contracts']['capabilities'][name] == expected
    assert full['audit']['model_projection']['omitted_input_schema_sha256_by_capability']
    normalized = [ActionCandidate.model_validate(value).model_dump(mode='json') for value in offered]
    assert [json.loads(value) for value in fake.calls[0]['kwargs']['option_descriptions'].values()] == normalized
    assert ('\nCandidate operations:\n' in fake.calls[0]['messages'][0]['content']) == (kind == 'hf')
    assert source == before


@pytest.mark.parametrize('receipt_value', [0, False, 0.0])
def test_v4_observation_references_require_exact_numeric_types(receipt_value):
    source = context()
    current = source['benchmark_public_context']['current_public_view']['observation']
    current['observations'] = {'sample': {'value': 0.0, 'quality': 'valid', 'time_s': 0.0}}
    action = {'kind': 'wait', 'dt_s': 1.0}
    source['benchmark_public_context']['complete_native_history'] = [
        {'proposed_action': deepcopy(action), 'executed_action': deepcopy(action),
         'response': {'observations': {'sample': {'value': receipt_value, 'quality': 'valid', 'time_s': 0.0}}}}]
    chooser = LocalDecisionBackend({'kind': 'native', 'context_protocol': 'decision-view-v4'}, backend=CapturingChooser())
    chooser.choose(source, candidates())
    model = chooser.last_call['public_input']['model_decision_view']
    receipt = model['history']['receipts'][0]['receipt']
    assert receipt['proposed_action_ref'] == 'executed_action'
    assert receipt['executed_action'] == action
    observations = receipt['response']['observations']
    assert ('$decision_ref' in observations) == (type(receipt_value) is float)
    if type(receipt_value) is not float:
        assert type(observations['sample']['value']) is type(receipt_value)


def test_v4_router_preserves_exact_routes_goal_and_native_state():
    source = context()
    chooser = LocalDecisionBackend({'kind': 'native', 'context_protocol': 'decision-view-v4'}, backend=CapturingChooser())
    chooser.decide(source, ['diagnose', 'verify'])
    metadata = chooser.last_call['public_input']
    assert metadata['model_decision_view']['allowed_routes'] == ['diagnose', 'verify']
    assert metadata['model_decision_view']['state'] == metadata['decision_view']['state']
