"""Decision-content projection preserves all native evidence and exact choices."""
from copy import deepcopy
import json

import pytest

from graph_local.backends import LocalDecisionBackend
from graph_local.decision_context import decision_model_content
from test_decision_context_v2 import CapturingChooser, candidates, context


@pytest.mark.parametrize('kind', ['hf', 'native'])
def test_v3_preserves_native_state_and_choices_while_recording_both_views(kind):
    source = context()
    current = source['benchmark_public_context']['current_public_view']['observation']
    current['source_sha256'] = 'native data with this name must remain visible'
    receipt = {'executed_action': {'kind': 'wait', 'dt_s': 30},
               'proposed_action': {'kind': 'wait', 'dt_s': 30},
               'response': {'status': 'succeeded', 'sample_time_s': 25,
                            'source_sha256': 'native receipt field must remain visible'}}
    source['benchmark_public_context']['complete_native_history'] = [receipt]
    original = deepcopy(source)
    offered = candidates()
    fake = CapturingChooser()
    chooser = LocalDecisionBackend({'kind': kind, 'context_protocol': 'decision-view-v3'}, backend=fake)
    chooser.choose(source, offered)
    metadata = chooser.last_call['public_input']
    full, model = metadata['decision_view'], metadata['model_decision_view']
    assert full['audit']['input_sha256']
    assert 'audit' not in model
    assert model['state'] == full['state']
    assert model['state']['current_observation'] == current
    assert model['history']['receipts'][0]['receipt'] == full['history']['receipts'][0]['receipt']
    assert model['history']['receipts'][0]['receipt']['response'] == receipt['response']
    assert model['candidate_contracts']['capabilities'] == full['candidate_contracts']['capabilities']
    assert model['candidate_contracts']['state_slots'] == full['candidate_contracts']['state_slots']
    actual_text = fake.calls[0]['messages'][0]['content']
    assert json.dumps(model, sort_keys=True, ensure_ascii=False, separators=(',', ':')) in actual_text
    criteria = fake.calls[0]['kwargs']['option_descriptions']
    assert [json.loads(value)['intent']['arguments']['action'] for value in criteria.values()] == [
        value['intent']['arguments']['action'] for value in offered]
    assert ('\nCandidate operations:\n' in actual_text) == (kind == 'hf')
    assert metadata['context'] == original
    assert source == original


def test_v3_router_retains_current_goal_native_readings_and_routes():
    source = context()
    chooser = LocalDecisionBackend({'kind': 'native', 'context_protocol': 'decision-view-v3'},
                                   backend=CapturingChooser())
    chooser.decide(source, ['diagnose', 'verify'])
    metadata = chooser.last_call['public_input']
    full, model = metadata['decision_view'], metadata['model_decision_view']
    assert model['state'] == full['state']
    assert model['allowed_routes'] == ['diagnose', 'verify']
    assert full['audit']['input_sha256']
    assert 'audit' not in model
    assert 'model_context_policy' in chooser.manifest()


def test_content_projection_never_changes_recorded_full_view():
    full = {'state': {'quality': 'estimated', 'sample_time_s': 0},
            'history': {'sha256': 'audit', 'receipts': [
                {'receipt': {'response': {'sha256': 'actual-native-value'}},
                 'projected_response_fields': {'contract': {'sha256': 'audit-only'}}}]},
            'candidate_contracts': {'capabilities': {'sample': {'source_sha256': 'public-schema-value'}},
                                    'source_sha256': 'audit'},
            'audit': {'input_sha256': 'audit'}}
    original = deepcopy(full)
    model = decision_model_content(full)
    assert full == original
    assert model['history']['receipts'][0]['receipt']['response']['sha256'] == 'actual-native-value'
    assert model['candidate_contracts']['capabilities']['sample']['source_sha256'] == 'public-schema-value'
