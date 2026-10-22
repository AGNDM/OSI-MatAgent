"""Online next-action selection, handoff safety and compatibility; no benchmark suite."""
from copy import deepcopy
import json
import math
import time

import httpx
import pytest

from app.graph import build_graph, initial_state
from app.lab_adapter import LabAdapter
from app.models import Limits, OperationDecision, OperationOptions
from app.operation_decision import LocalOperationBackend, ScriptedOperationBackend, validate_decision
from app.roles import ScriptedRoles
from app.scenarios import case_by_id
from app.trace import Trace
from bench.run import run_episode


def run(backend, roles=None, limits=None, state_update=None):
    lab = LabAdapter(case_by_id('dev-01'))
    trace = Trace('operation-test')
    limits = limits or Limits()
    state = initial_state(lab, trace.episode_id, limits)
    if state_update:
        state_update(lab, state)
    final = build_graph(lab, trace, limits, roles=roles or ScriptedRoles(), operation_backend=backend).invoke(
        state, {'configurable': {'thread_id': trace.episode_id}, 'recursion_limit': limits.graph_steps})
    return lab, trace, final


@pytest.mark.parametrize('case_id', ['dev-01', 'dev-02', 'dev-03', 'dev-04'])
def test_new_node_preserves_drying_paths_and_unresolved_case(case_id):
    record = run_episode(case_by_id(case_id), operation_backend=ScriptedOperationBackend())
    assert record['score']['task_success']
    assert record['report']['operation_call_count'] == record['report']['action_count']
    assert record['report']['llm_call_count'] == 0


def test_selected_candidate_precedes_procedure_and_tool_gate():
    lab, trace, final = run(ScriptedOperationBackend())
    kinds = [(e['component'], e['kind']) for e in trace.events]
    assert kinds.index(('diagnostic', 'role_output')) < kinds.index(('operation_decision', 'operation_selected'))
    assert kinds.index(('operation_decision', 'operation_selected')) < kinds.index(('procedure', 'role_start'))
    assert kinds.index(('gate', 'accepted')) < kinds.index(('execute', 'tool_result'))
    selected = [e for e in trace.events if e['kind'] == 'operation_selected']
    assert any(len(e['candidates']) > 1 for e in selected)
    for event in selected:
        matching = next(c for c in event['candidates'] if c['candidate_id'] == event['decision']['choice'])
        assert event['selected'] == matching
        encoded = json.dumps(event['input'])
        assert 'truth.process' not in encoded and 'initial_liquid_kg' not in encoded and '"family"' not in encoded
    assert final['report']['status'] == 'supported'


class BadDecision:
    backend_id = 'bad-operation-fixture'
    is_llm = True
    last_call = None

    def choose(self, context, candidates):
        return OperationDecision(backend_id=self.backend_id, choice='invented', probabilities={'invented': 1})


def test_invalid_selection_stops_without_tools_or_silent_fallback():
    lab, trace, final = run(BadDecision())
    assert lab.action_count == 0
    assert final['report']['terminal_reason'] == 'operation_decision_error'
    assert final['report']['llm_call_count'] == 1
    assert final['report']['operation_call_count'] == 1
    assert not any(e['component'] == 'procedure' for e in trace.events)


def test_operation_inference_obeys_model_budget_before_sending():
    lab, _, final = run(BadDecision(), limits=Limits(llm_calls=0))
    assert lab.action_count == 0
    assert final['report']['terminal_reason'] == 'llm_budget'
    assert final['report']['operation_call_count'] == 0


def test_operation_inference_obeys_deadline_after_result():
    class Slow(ScriptedOperationBackend):
        is_llm = True
        def choose(self, context, candidates):
            self.deadline = 0
            time.sleep(.02)
            return super().choose(context, candidates)
    lab, _, final = run(Slow(), limits=Limits(episode_seconds=.01))
    assert lab.action_count == 0
    assert final['report']['terminal_reason'] == 'deadline'


def test_backend_cannot_change_candidate_arguments_in_place():
    class Mutating(ScriptedOperationBackend):
        def choose(self, context, candidates):
            result = super().choose(context, candidates)
            candidates[0]['intent']['arguments'] = {'injected': True}
            return result
    lab, _, final = run(Mutating())
    assert final['report']['status'] == 'supported'
    assert lab.history[0]['proposal']['arguments'] == {}


def test_world_revision_change_during_inference_invalidates_selection():
    lab = LabAdapter(case_by_id('dev-01'))
    class Stale(ScriptedOperationBackend):
        def choose(self, context, candidates):
            lab.revision += 1
            return super().choose(context, candidates)
    trace = Trace('stale-operation')
    limits = Limits()
    final = build_graph(lab, trace, limits, operation_backend=Stale()).invoke(
        initial_state(lab, trace.episode_id, limits),
        {'configurable': {'thread_id': trace.episode_id}})
    assert lab.action_count == 0 and final['report']['terminal_reason'] == 'operation_decision_error'


@pytest.mark.parametrize('defect', ['unknown_capability', 'unknown_evidence', 'bad_arguments', 'duplicate'])
def test_invalid_diagnostic_candidates_are_rejected_before_selection(defect):
    class Broken(ScriptedRoles):
        def invoke(self, role, prompt, payload, schema):
            value = super().invoke(role, prompt, payload, schema)
            if role == 'diagnostic':
                c = value['candidates'][0]
                if defect == 'unknown_capability': c['intent']['capability_id'] = 'truth.inspect'
                elif defect == 'unknown_evidence': c['evidence_ids'] = ['invented-observation']
                elif defect == 'bad_arguments': c['intent']['arguments'] = {'bad': True}
                else: value['candidates'].append(deepcopy(c))
            return value
    lab, _, final = run(ScriptedOperationBackend(), roles=Broken())
    assert lab.action_count == 0
    assert final['report']['terminal_reason'] == 'role_error'
    assert final['report']['operation_call_count'] == 0


def test_procedure_cannot_replace_selected_operation():
    class Swapping(ScriptedRoles):
        def invoke(self, role, prompt, payload, schema):
            value = super().invoke(role, prompt, payload, schema)
            if role == 'procedure':
                value['capability_id'] = 'balance.clean'
            return value
    lab, _, final = run(ScriptedOperationBackend(), roles=Swapping())
    assert lab.action_count == 0 and final['report']['terminal_reason'] == 'role_error'


@pytest.mark.parametrize('probabilities,choice', [({'a': .8}, 'a'), ({'a': -.1, 'b': 1.1}, 'b'),
                                                ({'a': .7, 'b': .3}, 'b')])
def test_probability_contract_rejects_missing_candidates_invalid_values_or_wrong_argmax(probabilities, choice):
    candidates = [{'candidate_id': k, 'intent': {'capability_id': 'workflow.inspect', 'arguments': {},
        'postcondition': 'Inspect.', 'discriminating_observation': 'Read public state.'},
        'rationale': 'Check evidence.', 'evidence_ids': []} for k in ['a', 'b']]
    with pytest.raises(ValueError):
        validate_decision(OperationDecision(backend_id='fixture', choice=choice, probabilities=probabilities), candidates)


def test_local_operation_adapter_uses_general_candidates_and_actual_token_probabilities(monkeypatch):
    contexts = []
    def respond(request):
        body = json.loads(request.content)
        if request.url.path == '/apply-template':
            contexts.append(body['messages'][0]['content'])
            return httpx.Response(200, json={'prompt': 'formatted'})
        assert body['n_predict'] == 1 and body['grammar'] == 'root ::= [AB]'
        return httpx.Response(200, json={'probs': [{'top_probs': [
            {'token': 'A', 'prob': .2}, {'token': 'B', 'prob': .6}, {'token': 'X', 'prob': .2}]}]})
    client = httpx.AsyncClient
    monkeypatch.setattr('app.local_decision.httpx.AsyncClient',
                        lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    backend = LocalOperationBackend()
    candidates = [{'candidate_id': 'heat', 'intent': {'capability_id': 'thermal.prepare_reference'}},
                  {'candidate_id': 'scan', 'intent': {'capability_id': 'xrd.acquire', 'arguments': {'stop_deg': 60}}}]
    decision = backend.choose({'target': 'titania'}, candidates)
    assert decision.choice == 'scan' and decision.probabilities['scan'] == pytest.approx(.75)
    assert backend.last_call['option_probability_mass'] == pytest.approx(.8)
    assert 'Which specialist' not in contexts[0] and 'xrd.acquire' in contexts[0]


def test_action_budget_exhaustion_blocks_operational_model_call():
    lab, _, final = run(BadDecision(), limits=Limits(lab_actions=0))
    assert lab.action_count == 0
    assert final['report']['terminal_reason'] == 'action_budget'
    assert final['report']['operation_call_count'] == 0


def test_gate_enforces_selection_even_if_profile_skips_procedure_validation():
    from app.roles import PROMPTS, projection as public_projection
    class PermissiveProfile:
        prompts = PROMPTS
        projection = staticmethod(public_projection)
        retrieve = staticmethod(lambda role: [])
        validate_role_contract = staticmethod(lambda *args: None)
    class Swap(ScriptedRoles):
        is_llm = True
        def invoke(self, role, prompt, payload, schema):
            value = super().invoke(role, prompt, payload, schema)
            if role == 'procedure': value['capability_id'] = 'balance.clean'
            return value
    lab = LabAdapter(case_by_id('dev-01'))
    trace = Trace('gate-selection-test')
    limits = Limits(llm_calls=3)
    final = build_graph(lab, trace, limits, roles=Swap(), profile=PermissiveProfile(),
                        operation_backend=ScriptedOperationBackend()).invoke(
        initial_state(lab, trace.episode_id, limits), {'configurable': {'thread_id': trace.episode_id}})
    assert lab.action_count == 0
    assert any(e['kind'] == 'rejected' for e in trace.events)
    assert final['report']['terminal_reason'] == 'llm_budget'


def test_ui_operation_mode_exposes_choices_and_preserves_action_replay(tmp_path):
    from app.demo import DemoController
    controller = DemoController(tmp_path)
    key = controller.start('dev-01', 'scripted-operation-rules')
    end = time.monotonic()+10
    while controller.state()['running'] and time.monotonic() < end:
        time.sleep(.01)
    state = controller.state()
    assert not state['running'] and state['record']['score']['task_success']
    assert any(e['kind'] == 'operation_selected' for e in state['events'])
    assert all('input' not in e and 'llm_metadata' not in e for e in state['events'])
    assert controller.replay(key)['matched_actions'] == 3


def test_titania_diagnosis_cannot_restart_initial_inspection_after_current_readback():
    from app.models import HypothesisSet, Intent
    from app.titania import TitaniaProfile
    record = HypothesisSet(hypotheses=[], intent=Intent(capability_id='workflow.inspect', arguments={},
        discriminating_observation='Reinspect.', postcondition='Get state.'))
    payload = {'observations': {'revision': 0, 'inspection': {'id': 'obs-1', 'revision': 0, 'quality': 'valid'}},
               'capabilities': {'workflow.inspect': {'input_schema': {'type': 'object', 'properties': {}}}},
               'last_error': None}
    with pytest.raises(ValueError, match='already valid'):
        TitaniaProfile.validate_role_contract('diagnostic', record, payload)
    payload['observations']['revision'] = 1
    TitaniaProfile.validate_role_contract('diagnostic', record, payload)


def test_run_summary_preserves_operation_backend_and_separates_legacy_records():
    from bench.run import summarize_runs
    legacy = run_episode(case_by_id('dev-03'))
    enabled = run_episode(case_by_id('dev-03'), operation_backend=ScriptedOperationBackend())
    # Older recording formats omit this field entirely.
    legacy.pop('operation_backend')
    rows = [r for r in summarize_runs([legacy, enabled]) if r['family'] == 'overall']
    assert len(rows) == 2
    assert {r['operation_backend'] for r in rows} == {None, ScriptedOperationBackend.backend_id}
    assert all(r['attempted'] == 1 for r in rows)
