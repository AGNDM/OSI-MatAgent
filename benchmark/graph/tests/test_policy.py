"""Actual application graph, using explicit offline test-double backends.

These tests verify bridge behavior, not language-model performance.
"""
from copy import deepcopy
import os
import unittest

from app.models import Limits, OperationDecision, RouteDecision
from graph_local.policy import GraphNativePolicy, PublicBenchmarkLab, GraphPolicyError


class Menus:
    def action_options(self, view, history):
        options = [{'text': name, 'action': {'kind': 'select_instrument', 'instrument_id': name}}
                   for name in ('machine-a', 'machine-b')]
        if view.get('selected_instrument'):
            options.append({'text': 'Read native state', 'capability_id': '__read__'})
        options.extend([{'text': 'Finish', 'action': {'kind': 'finish'}},
                        {'text': 'Escalate', 'action': {'kind': 'escalate'}}])
        return options

    def argument_options(self, view, history, option):
        if 'action' in option:
            return [deepcopy(option)]
        return [{'text': 'Read all', 'action': {'kind': 'read', 'instrument_id': view['selected_instrument'], 'state_ids': None}},
                {'text': 'Read explicit', 'action': {'kind': 'read', 'instrument_id': view['selected_instrument'], 'state_ids': ['power.on']}}]


class OfflineRoles:
    backend_id = 'offline-test-fixture-expert'
    is_llm = False

    def __init__(self, actions, verdict='supported'):
        self.actions = iter(actions)
        self.payloads = []
        self.verdict = verdict

    def invoke(self, role, prompt, payload, schema):
        self.payloads.append((role, deepcopy(payload)))
        if role == 'diagnostic':
            action = next(self.actions)
            return {'hypotheses': [], 'candidates': [{'candidate_id': 'fixture-candidate',
                    'intent': {'capability_id': 'benchmark.' + action['kind'], 'arguments': {'action': action},
                               'discriminating_observation': 'Offline bridge contract test', 'postcondition': 'Native public evidence'},
                    'rationale': 'Offline fixture only', 'evidence_ids': [], 'material_evidence_ids': []}]}
        if role == 'procedure':
            return {'sample_id': payload['sample_id'], 'expected_revision': payload['expected_revision'],
                    'request_id': payload['request_id'], 'capability_id': payload['intent']['capability_id'],
                    'arguments': deepcopy(payload['intent']['arguments']), 'postcondition': payload['intent']['postcondition']}
        if role == 'verification':
            return {'status': self.verdict, 'evidence_ids': [payload['observations']['measurement']['id']], 'missing_evidence': []}
        raise AssertionError(role)


class OfflineChooser:
    backend_id = 'offline-test-fixture-chooser'
    is_llm = False
    is_decision_model = False
    allow_rules_fallback = False

    def decide(self, summary, allowed):
        choice = 'prepare_action' if summary['has_intent'] else 'diagnose'
        return RouteDecision(backend_id=self.backend_id, choice=choice,
                             probabilities={key: float(key == choice) for key in allowed})

    def choose(self, context, candidates):
        choice = candidates[0]['candidate_id']
        return OperationDecision(backend_id=self.backend_id, choice=choice,
                                 probabilities={candidate['candidate_id']: float(candidate['candidate_id'] == choice) for candidate in candidates})


def view(selected=None):
    return {'goal': 'Offline bridge test goal', 'selected_instrument': selected,
            'manual': [{'text': 'Complete public test manual', 'id': 'manual'}],
            'observation': {'power.on': {'value': True, 'unit': '1', 'quality': 'valid', 'sample_time_s': 7}},
            'budget': {'max_decisions': 16, 'max_sim_time_s': 100}, 'contract': {}}


def ack(proposed, executed=None, corrected=False):
    return {'proposed_action': deepcopy(proposed), 'executed_action': deepcopy(executed or proposed),
            'corrected': corrected, 'response': {'status': 'succeeded', 'native_marker': 'actual-test-ack'}}


@unittest.skipUnless(os.environ.get('SLURMD_NODENAME') and os.environ.get('SLURM_JOB_ID'), 'Requires genuine Slurm allocation')
class ActualGraphBridgeTests(unittest.TestCase):
    def policy(self, actions, mode='free_run', verdict='supported'):
        roles = OfflineRoles(actions, verdict)
        task = {'task_id': 'public-fixture-only', 'initial_context': view()}
        policy = GraphNativePolicy(task, Menus(), roles, OfflineChooser(), mode=mode,
            limits=Limits(lab_actions=16, llm_calls=64, episode_seconds=60, graph_steps=128))
        return policy, roles

    def test_actual_graph_gate_handoff_and_persistent_public_history(self):
        select = {'kind': 'select_instrument', 'instrument_id': 'machine-a'}
        read = {'kind': 'read', 'instrument_id': 'machine-a', 'state_ids': None}
        policy, roles = self.policy([select, read, {'kind': 'finish'}])
        self.assertEqual(policy.next_action(view(), []), select)
        self.assertEqual(policy.lab.revision, 0)  # No optimistic execution.
        history = [ack(select)]
        self.assertEqual(policy.next_action(view('machine-a'), history), read)
        self.assertEqual(policy.lab.revision, 1)
        history.append(ack(read))
        self.assertEqual(policy.next_action(view('machine-a'), history), {'kind': 'finish'})
        diagnostics = [payload for role, payload in roles.payloads if role == 'diagnostic']
        self.assertEqual([len(p['benchmark_public_context']['complete_native_history']) for p in diagnostics], [0, 1, 2])
        self.assertEqual(diagnostics[-1]['benchmark_public_context']['current_public_view']['observation']['power.on']['sample_time_s'], 7)
        self.assertTrue(any(event['kind'] == 'accepted' for event in policy.trace.events))
        self.assertEqual(policy.graph.get_state(policy.config).values['terminal_reason'], 'agent_supported')

    def test_teacher_forcing_acknowledges_actual_replacement(self):
        wrong = {'kind': 'select_instrument', 'instrument_id': 'machine-b'}
        actual = {'kind': 'select_instrument', 'instrument_id': 'machine-a'}
        read = {'kind': 'read', 'instrument_id': 'machine-a', 'state_ids': None}
        policy, roles = self.policy([wrong, read], mode='teacher_forced')
        self.assertEqual(policy.next_action(view(), []), wrong)
        self.assertEqual(policy.next_action(view('machine-a'), [ack(wrong, actual, True)]), read)
        contexts = [payload['benchmark_public_context'] for role, payload in roles.payloads if role == 'diagnostic']
        self.assertEqual(contexts[-1]['complete_native_history'][0]['executed_action'], actual)
        self.assertTrue(contexts[-1]['complete_native_history'][0]['corrected'])

    def test_free_run_rejects_correction(self):
        first = {'kind': 'select_instrument', 'instrument_id': 'machine-b'}
        policy, _ = self.policy([first])
        policy.next_action(view(), [])
        with self.assertRaisesRegex(RuntimeError, 'replacement'):
            policy.next_action(view('machine-a'), [ack(first, {'kind': 'select_instrument', 'instrument_id': 'machine-a'}, True)])

    def test_acknowledgement_must_match_exact_proposal(self):
        first = {'kind': 'select_instrument', 'instrument_id': 'machine-a'}
        policy, _ = self.policy([first])
        policy.next_action(view(), [])
        with self.assertRaisesRegex(RuntimeError, 'another proposed action'):
            policy.next_action(view('machine-a'), [ack({'kind': 'select_instrument', 'instrument_id': 'machine-b'})])

    def test_off_preset_action_rejected_without_native_execution(self):
        lab = PublicBenchmarkLab({'task_id': 'test'}, Menus(), view('machine-a'), [], 'free_run')
        with self.assertRaisesRegex(ValueError, 'absent'):
            lab.intent_action({'capability_id': 'benchmark.read', 'arguments': {'action': {
                'kind': 'read', 'instrument_id': 'machine-a', 'state_ids': ['private.truth']}}})

    def test_entire_finite_preset_bank_is_in_public_context(self):
        lab = PublicBenchmarkLab({'task_id': 'test'}, Menus(), view('machine-a'), [], 'free_run')
        context = lab.context()
        read = next(command for command in context['complete_offered_commands_and_presets'] if command['command'].get('capability_id') == '__read__')
        self.assertEqual(len(read['argument_presets']), 2)
        self.assertIn('Complete public test manual', json_text(context))

    def test_finish_requires_real_verifier_node(self):
        policy, roles = self.policy([{'kind': 'finish'}])
        self.assertEqual(policy.next_action(view('machine-a'), []), {'kind': 'finish'})
        self.assertEqual([role for role, payload in roles.payloads], ['diagnostic', 'procedure', 'verification'])

    def test_escalation_remains_native_action_not_forced_success(self):
        policy, _ = self.policy([{'kind': 'escalate'}])
        self.assertEqual(policy.next_action(view(), []), {'kind': 'escalate'})
        self.assertIsNone(policy.graph.get_state(policy.config).values.get('terminal_reason'))

    def test_premature_finish_correction_resumes_same_graph_and_preserves_budgets(self):
        read = {'kind': 'read', 'instrument_id': 'machine-a', 'state_ids': None}
        policy, roles = self.policy([{'kind': 'finish'}, read], mode='teacher_forced')
        roles.is_llm = True  # Simulate call-budget accounting with offline outputs.
        self.assertEqual(policy.next_action(view('machine-b'), []), {'kind': 'finish'})
        graph = policy.graph
        before = graph.get_state(policy.config).values
        calls, deadline = before['llm_call_count'], before['deadline']
        correction = ack({'kind': 'finish'}, {'kind': 'select_instrument', 'instrument_id': 'machine-a'}, True)
        self.assertEqual(policy.next_action(view('machine-a'), [correction]), read)
        after = graph.get_state(policy.config).values
        self.assertIs(policy.graph, graph)
        self.assertEqual(after['deadline'], deadline)
        self.assertGreater(after['llm_call_count'], calls)
        self.assertEqual(after['action_count'], 1)
        self.assertIsNone(after.get('terminal_reason'))
        diagnostic = [payload for role, payload in roles.payloads if role == 'diagnostic'][-1]
        self.assertEqual(diagnostic['benchmark_public_context']['complete_native_history'], [correction])

    def test_unknown_token_counts_remain_numeric_with_false_known_flags(self):
        policy, _ = self.policy([])
        policy.trace.emit('diagnostic', 'role_error', llm_metadata={'request_sent': True, 'usage': None})
        policy.trace.emit('router', 'routing', backend_metadata={'request_sent': True,
            'usage': {'prompt_tokens': 7, 'completion_tokens': 2}})
        usage = policy._usage(0)
        self.assertFalse(usage['input_tokens_known'])
        self.assertFalse(usage['output_tokens_known'])
        self.assertEqual(usage['input_tokens'], 7)
        self.assertEqual(usage['output_tokens'], 2)
        self.assertEqual(usage['graph_model_calls'], 2)

    def test_policy_failure_is_terminal_abort_signal_and_keeps_reason(self):
        policy, _ = self.policy([])
        with self.assertRaises(GraphPolicyError):
            policy.next_action(view(), [])
        self.assertIsNotNone(policy.terminal_reason)
        self.assertTrue(any(event['kind'] == 'policy_abort' for event in policy.trace.events))

    def test_verification_is_unavailable_until_actual_instrument_readings_exist(self):
        class ObservedChooser(OfflineChooser):
            def __init__(self):
                self.routes = []

            def decide(self, summary, allowed):
                self.routes.append(list(allowed))
                return super().decide(summary, allowed)

        select = {'kind': 'select_instrument', 'instrument_id': 'machine-a'}
        read = {'kind': 'read', 'instrument_id': 'machine-a', 'state_ids': None}
        policy, _ = self.policy([select, read])
        chooser = ObservedChooser()
        policy.decision_backend = chooser
        initial = view()
        initial['observation'] = {}
        self.assertEqual(policy.next_action(initial, []), select)
        self.assertNotIn('verify', chooser.routes[0])
        self.assertIsNone(policy.lab.observe()['measurement'])
        self.assertIn('no physical instrument reading', policy.lab.observe()['inspection']['evidence_scope'])
        old_route_count = len(chooser.routes)
        self.assertEqual(policy.next_action(view('machine-a'), [ack(select)]), read)
        self.assertIsNotNone(policy.lab.observe()['measurement'])
        self.assertTrue(any('verify' in allowed for allowed in chooser.routes[old_route_count:]))
        empty_native = view('machine-a')
        empty_native['observation'] = {'status': 'succeeded', 'observations': {}}
        empty_lab = PublicBenchmarkLab({'task_id': 'no-readings'}, Menus(), empty_native, [], 'free_run')
        self.assertIsNone(empty_lab.observe()['measurement'])


def json_text(value):
    import json
    return json.dumps(value)


if __name__ == '__main__':
    unittest.main()
