"""PRIVILEGED OFFLINE DIAGNOSTICS ONLY: real graph + real native benchmark.

The reference-backed expert below receives private author fixtures solely in
this test module. It is never used by graph_local.run or counted as a model.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
    pytest.skip('Native graph diagnostics require a genuine Slurm allocation', allow_module_level=True)

from app.models import Limits
from graph_local.policy import GraphNativePolicy
from test_policy import OfflineChooser, OfflineRoles


def native_fixture():
    """Use the already activated authenticated benchmark when available."""
    bootstrap = sys.modules.get('sequential_120.runtime.bootstrap')
    package = Path(__file__).resolve().parents[1]
    default = Path(bootstrap.__file__).resolve().parents[2] if bootstrap else next(
        (path for path in (package.parent / 'instrument', package / 'instrument')
         if (path / 'sequential_120/cases.jsonl').is_file()), None)
    selected = os.environ.get('GRAPH_LOCAL_BUNDLE', os.environ.get('COMPARISON_TEST_BUNDLE', default))
    if selected is None or not (Path(selected) / 'sequential_120/cases.jsonl').is_file():
        pytest.skip('Set GRAPH_LOCAL_BUNDLE to the authenticated instrument bundle, or use tools/check.py to activate it')
    bundle = Path(selected)
    public = {row['task_id']: row for row in map(json.loads, (bundle / 'sequential_120/cases.jsonl').read_text().splitlines())}
    private = [json.loads(line) for line in (bundle / 'sequential_120/evaluator/case-specs.jsonl').read_text().splitlines()]
    from stepwise_choice.engine import MenuBuilder, NativeReferencePolicy, run_case
    menus = MenuBuilder(json.loads((bundle / 'sequential_120/action-presets.json').read_text()))
    return public, private, menus, NativeReferencePolicy, run_case


class PrivilegedReferenceExpert(OfflineRoles):
    backend_id = 'PRIVILEGED-OFFLINE-native-reference-graph-integration-diagnostic'
    is_llm = False

    def __init__(self, reference, force_early_finish=False):
        super().__init__([])
        self.reference = reference
        self.force_early_finish = force_early_finish
        self.injected = False

    def invoke(self, role, prompt, payload, schema):
        if role != 'diagnostic':
            return super().invoke(role, prompt, payload, schema)
        self.payloads.append((role, deepcopy(payload)))
        context = payload['benchmark_public_context']
        history = context['complete_native_history']
        if self.force_early_finish and len(history) == 1 and not self.injected:
            action = {'kind': 'finish'}
            self.injected = True
        else:
            action = self.reference.next_action(context['current_public_view'], history)
        return {'hypotheses': [], 'candidates': [{
            'candidate_id': 'privileged-test-reference',
            'intent': {'capability_id': 'benchmark.' + action['kind'], 'arguments': {'action': deepcopy(action)},
                       'discriminating_observation': 'PRIVILEGED author diagnostic, not a model inference',
                       'postcondition': 'Unchanged native author rubric'},
            'rationale': 'Private reference is permitted only inside this explicitly privileged offline test',
            'evidence_ids': [], 'material_evidence_ids': []}]}


def diagnostic_policy(public, case, menus, reference_type, mode, force_early_finish=False):
    expert = PrivilegedReferenceExpert(reference_type(case), force_early_finish)
    policy = GraphNativePolicy(deepcopy(public), menus, expert, OfflineChooser(), mode=mode,
        limits=Limits(lab_actions=public['initial_context']['budget']['max_decisions'],
                      llm_calls=256, episode_seconds=300, graph_steps=2048))
    policy.privileged = True
    policy.identity.update(privileged=True, diagnostic_only=True, model_performance=False)
    return policy, expert


def test_fourteen_workflows_both_modes_through_actual_graph_and_native_grader():
    public, private, menus, reference_type, run_case = native_fixture()
    representatives = {}
    for case in private:
        representatives.setdefault(case['workflow_id'], case)
    assert len(representatives) == 14
    assert len({case['instrument_id'] for case in representatives.values()}) == 8
    for workflow, case in representatives.items():
        for mode in ('teacher_forced', 'free_run'):
            policy, expert = diagnostic_policy(public[case['task_id']], case, menus, reference_type, mode)
            episode = run_case(case, policy, mode=mode, menu_builder=menus)
            assert episode['trajectory_outcome_met'], (workflow, mode, episode.get('termination'), episode.get('decision_records', [])[-1:])
            assert episode['menu_membership_verified'] is True
            assert episode['interventions'] == 0
            assert expert.payloads
            assert policy.identity['diagnostic_only'] is True
            assert policy.identity['model_performance'] is False


def test_actual_native_teacher_forced_premature_finish_resumes_graph():
    public, private, menus, reference_type, run_case = native_fixture()
    case = next(case for case in private if case['workflow_id'] == 'furnace_program') if any(
        case['workflow_id'] == 'furnace_program' for case in private) else next(
        case for case in private if case['instrument_id'] == 'nabertherm_l9')
    policy, expert = diagnostic_policy(public[case['task_id']], case, menus, reference_type,
                                       'teacher_forced', force_early_finish=True)
    episode = run_case(case, policy, mode='teacher_forced', menu_builder=menus)
    assert expert.injected
    assert episode['trajectory_outcome_met']
    assert episode['interventions'] >= 1
    finish_corrections = [record for record in episode['decision_records']
                          if record['proposed_action']['kind'] == 'finish' and record['corrected']]
    assert finish_corrections
    payloads = [payload for role, payload in expert.payloads if role == 'diagnostic']
    assert any(any(item['proposed_action']['kind'] == 'finish' and item['corrected']
                   for item in payload['benchmark_public_context']['complete_native_history']) for payload in payloads)
    assert policy.lab.history
