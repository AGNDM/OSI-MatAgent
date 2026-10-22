"""Persistent actual LangGraph policy with acknowledged external native handoffs.

The benchmark alone executes actions and grades them. The application graph is
paused after its typed gate and resumed only after matching native feedback.
"""
from copy import deepcopy
import hashlib
import json
import time
from uuid import uuid4

from app.graph import build_graph, initial_state
from app.lab_adapter import GateRejected
from app.models import Limits
from app.trace import Trace
from graph_local.profile import BenchmarkProfile


KINDS = ('select_instrument', 'describe', 'invoke', 'read', 'wait', 'finish', 'escalate')


class GraphPolicyError(RuntimeError):
    """Terminal graph/bridge failure; runner must abort, never teacher-correct."""


def canonical(action, selected=None):
    if not isinstance(action, dict) or action.get('kind') not in KINDS:
        raise ValueError('Invalid public native action')
    kind = action['kind']
    expected = {'kind', 'instrument_id'} | {'invoke': {'capability_id', 'arguments'}, 'read': {'state_ids'}, 'wait': {'dt_s'}}.get(kind, set())
    if set(action) - expected:
        raise ValueError('Non-public native action fields')
    if kind in ('finish', 'escalate'):
        return {'kind': kind}
    if kind == 'wait':
        return {'kind': kind, 'dt_s': action.get('dt_s')}
    value = {'kind': kind, 'instrument_id': action.get('instrument_id', selected)}
    if kind == 'invoke':
        value.update(capability_id=action.get('capability_id'), arguments=deepcopy(action.get('arguments', {})))
    elif kind == 'read':
        slots = action.get('state_ids')
        value['state_ids'] = sorted(slots) if isinstance(slots, list) else slots
    return value


class PublicBenchmarkLab:
    routing_policy = 'argmax-v1'
    router_wait_enabled = False
    count_router_calls = True

    def __init__(self, public_task, menus, view, history, mode):
        self.task = deepcopy(public_task)
        self.menus, self.mode = menus, mode
        self.sample_id = 'benchmark:' + str(public_task['task_id'])
        self.view, self.history = deepcopy(view), deepcopy(history)
        self.revision = self.action_count = len(history)
        self.last_mutation_step = self.revision
        self.queued = None

    def target(self):
        return {'sample_id': self.sample_id, 'goal': self.view.get('goal', self.task.get('goal')),
                'source': 'Public benchmark goal; native evaluator retains official scoring authority.'}

    def capabilities(self):
        return {'benchmark.' + kind: {'arguments': {'action': 'Exactly one complete offered native action JSON'},
                 'source': 'Public native menu; full presets are supplied in benchmark_public_context.'} for kind in KINDS}

    def context(self):
        commands = []
        for option in self.menus.action_options(deepcopy(self.view), deepcopy(self.history)):
            commands.append({'command': deepcopy(option), 'argument_presets': deepcopy(
                self.menus.argument_options(deepcopy(self.view), deepcopy(self.history), deepcopy(option)))})
        return {'public_task': deepcopy(self.task), 'current_public_view': deepcopy(self.view),
                'complete_native_history': deepcopy(self.history), 'complete_offered_commands_and_presets': commands,
                'wrapper_revision': self.revision,
                'evidence_note': 'Cached native readings retain original units, quality and timestamps. Wrapper revision is transport state, not a native measurement timestamp.'}

    def observe(self):
        public_observation = self.view.get('observation') or {}
        readings = public_observation.get('observations') if 'observations' in public_observation else {
            key: value for key, value in public_observation.items()
            if isinstance(value, dict) and any(field in value for field in ('value', 'quality', 'unit'))}
        has_native_readings = bool(self.view.get('selected_instrument') and readings)
        evidence = {'id': self.sample_id + ':public:' + str(self.revision), 'sample_id': self.sample_id,
                    'revision': self.revision, 'captured_at_step': self.revision + 1,
                    'quality': 'valid', 'quality_semantics': 'Authenticated public snapshot transport only',
                    'context_reference': 'benchmark_public_context (supplied in full to every role/router/chooser)',
                    'evidence_scope': 'Native cached readings' if has_native_readings else 'Catalog and public context only; no physical instrument reading'}
        return {'revision': self.revision, 'sample_id': self.sample_id, 'inspection': deepcopy(evidence),
                'measurement': deepcopy(evidence) if has_native_readings else None, 'last_mutation_step': self.revision}

    def intent_action(self, intent):
        cap = intent.capability_id if hasattr(intent, 'capability_id') else intent['capability_id']
        arguments = intent.arguments if hasattr(intent, 'arguments') else intent['arguments']
        if not isinstance(arguments, dict) or set(arguments) != {'action'}:
            raise ValueError('Intent arguments must contain exactly action')
        action = arguments['action']
        if cap != 'benchmark.' + str(action.get('kind')):
            raise ValueError('Capability alias and native action kind disagree')
        self.check_offered(action)
        return deepcopy(action)

    def check_offered(self, action):
        target = canonical(action, self.view.get('selected_instrument'))
        for option in self.menus.action_options(deepcopy(self.view), deepcopy(self.history)):
            for preset in self.menus.argument_options(deepcopy(self.view), deepcopy(self.history), deepcopy(option)):
                if canonical(preset['action'], self.view.get('selected_instrument')) == target:
                    return
        raise ValueError('Action is absent from the complete public native menu')

    def validate(self, proposal, pending_operation=None):
        if pending_operation:
            raise GateRejected('PENDING_OPERATION_UNRESOLVED')
        if proposal.sample_id != self.sample_id or proposal.expected_revision != self.revision:
            raise GateRejected('STALE_PROPOSAL')
        self.intent_action(proposal)

    def route_context(self, state, summary, allowed):
        result = {**deepcopy(summary), 'benchmark_public_context': self.context()}
        if getattr(self, 'decision_context_protocol', None) in ('decision-view-v1', 'decision-view-v2', 'decision-view-v3', 'decision-view-v4'):
            result.update({key: deepcopy(state.get(key)) for key in
                ('observations', 'target', 'intent', 'hypotheses', 'last_error',
                 'verification', 'pending_operation', 'action_count', 'llm_call_count')})
        return result, list(allowed)

    def operation_context(self, context, candidates):
        return {**deepcopy(context), 'benchmark_public_context': self.context()}

    def verification_signature(self, state):
        return hashlib.sha256(json.dumps(self.context(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def queue_ack(self, view, history, proposed):
        if len(history) != len(self.history) + 1 or history[:-1] != self.history:
            raise RuntimeError('Native public history prefix or handoff count changed')
        ack = history[-1]
        selected = self.view.get('selected_instrument')
        if canonical(ack.get('proposed_action'), selected) != canonical(proposed, selected):
            raise RuntimeError('Native acknowledgement refers to another proposed action')
        executed = ack.get('executed_action')
        changed = canonical(executed, selected) != canonical(proposed, selected)
        if changed and not (self.mode == 'teacher_forced' and ack.get('corrected') is True):
            raise RuntimeError('Unannounced or unauthorized native action replacement')
        if ack.get('corrected') and self.mode != 'teacher_forced':
            raise RuntimeError('Correction is forbidden in free-run mode')
        self.check_offered(executed)
        if not isinstance(ack.get('response'), dict):
            raise RuntimeError('Native acknowledgement is missing its actual response')
        self.queued = (deepcopy(view), deepcopy(history), deepcopy(ack))

    def execute(self, proposal):
        # Called by the real graph only AFTER the native evaluator has acted.
        self.validate(proposal)
        return self.commit_ack(self.intent_action(proposal))

    def commit_ack(self, proposed):
        """Commit one validated native acknowledgement, including corrected finish."""
        if self.queued is None:
            raise RuntimeError('Graph execution requires native acknowledgement')
        view, history, ack = self.queued
        if canonical(proposed, self.view.get('selected_instrument')) != canonical(ack['proposed_action'], self.view.get('selected_instrument')):
            raise RuntimeError('Queued acknowledgement does not match typed proposal')
        self.view, self.history = view, history
        self.revision += 1
        self.action_count = len(history)
        self.last_mutation_step = self.revision
        self.queued = None
        return {'result': deepcopy(ack['response']), 'native_handoff_acknowledgement': deepcopy(ack),
                'revision': self.revision, 'action_count': self.action_count}


class GraphNativePolicy:
    privileged = False

    def __init__(self, public_task, menu_builder, roles, decision_backend, *, mode, trace_path=None, limits=None):
        if mode not in ('teacher_forced', 'free_run'):
            raise ValueError('Unknown native evaluation mode')
        self.public_task, self.menu_builder = deepcopy(public_task), menu_builder
        self.roles, self.decision_backend, self.mode = roles, decision_backend, mode
        budget = public_task.get('initial_context', {}).get('budget', {})
        self.limits = limits or Limits(lab_actions=budget.get('max_decisions', 64), llm_calls=256,
                                       episode_seconds=1800, graph_steps=2048, format_repairs=1)
        self.trace = Trace(str(public_task['task_id']) + ':' + mode, trace_path)
        self.config = {'configurable': {'thread_id': str(uuid4())}, 'recursion_limit': self.limits.graph_steps}
        self.graph = self.lab = None
        self.pending = None
        self.terminal_handoff = False
        self.terminal_reason = None
        self.last_action_usage = {}
        self.identity = {'protocol': 'actual-repository-langgraph-native-bridge-v1',
            'application_graph': 'app.graph.build_graph', 'mode': mode,
            'expert_backend': roles.backend_id, 'decision_backend': decision_backend.backend_id,
            'limits': self.limits.model_dump(), 'native_execution': 'External evaluator acknowledged handoff',
            'completion': 'Actual graph verifier claim, independently scored by native evaluator'}
        self.identity.update(expert_record_protocol=getattr(roles, 'record_protocol', None),
                             decision_context_protocol=getattr(decision_backend, 'context_protocol', None))

    def _usage(self, start):
        calls = []
        for event in self.trace.events[start:]:
            if event['kind'] not in ('role_output', 'role_error', 'routing', 'operation_selected', 'operation_error'):
                continue
            metadata = event.get('llm_metadata') or event.get('backend_metadata')
            if metadata and metadata.get('request_sent'):
                calls.append(metadata)
        usages = [call.get('usage') or {} for call in calls]
        inputs, outputs = [u.get('prompt_tokens') for u in usages], [u.get('completion_tokens') for u in usages]
        known_in = all(type(count) is int for count in inputs)
        known_out = all(type(count) is int for count in outputs)
        return {'input_tokens_known': known_in, 'output_tokens_known': known_out,
                # Native engine sums numeric counters even when unknown flags
                # are false. Zero here is a placeholder, never known usage.
                'input_tokens': sum(count for count in inputs if type(count) is int),
                'output_tokens': sum(count for count in outputs if type(count) is int),
                'gpu_seconds': None, 'api_cost_usd': 0., 'graph_model_calls': len(calls),
                'selection_events': [], 'protocol': self.identity['protocol']}

    def next_action(self, view, history):
        started = len(self.trace.events)
        try:
            if self.graph is None:
                self.lab = PublicBenchmarkLab(self.public_task, self.menu_builder, view, history, self.mode)
                self.lab.decision_context_protocol = getattr(self.decision_backend, 'context_protocol', None)
                self.graph = build_graph(self.lab, self.trace, self.limits, roles=self.roles,
                    decision_backend=self.decision_backend, profile=BenchmarkProfile(
                        self.lab, record_protocol=getattr(self.roles, 'record_protocol', None)),
                    operation_backend=self.decision_backend)
                value = initial_state(self.lab, self.trace.episode_id, self.limits)
            else:
                if self.pending is None:
                    raise RuntimeError('Terminal graph cannot be asked for another native action')
                self.lab.queue_ack(view, history, self.pending)
                if self.terminal_handoff:
                    if self.mode != 'teacher_forced' or history[-1].get('corrected') is not True:
                        raise RuntimeError('Only an explicitly corrected teacher-forced finish may resume a terminal graph')
                    self.lab.commit_ack(self.pending)
                    # Resume the SAME compiled graph and checkpoint. All model
                    # counters, previous hypotheses and original deadline stay.
                    self.graph.update_state(self.config, {'terminal_reason': None, 'report': {},
                        'verification': None, 'intent': None, 'proposal': None, 'candidates': [],
                        'operation_decision': None, 'operation_revision': None, 'router_wait_authorization': None,
                        'action_count': self.lab.action_count,
                        'last_error': {'code': 'NATIVE_FINISH_CORRECTED', 'native_feedback': deepcopy(history[-1])}},
                        as_node='execute')
                    self.terminal_handoff = False
                    self.terminal_reason = None
                value = None
            self.pending = None
            while True:
                self.graph.invoke(value, self.config, interrupt_before=['execute'])
                snapshot = self.graph.get_state(self.config)
                state = snapshot.values
                if state.get('terminal_reason'):
                    self.terminal_reason = state['terminal_reason']
                    if state['terminal_reason'] == 'agent_supported':
                        action = {'kind': 'finish'}
                        self.lab.check_offered(action)
                        if self.mode == 'teacher_forced':
                            self.pending = deepcopy(action)
                            self.terminal_handoff = True
                        return action
                    raise RuntimeError('Actual LangGraph stopped: ' + str(state['terminal_reason']))
                if snapshot.next != ('execute',):
                    raise RuntimeError('Graph did not reach its typed native handoff')
                from app.models import ActionProposal
                proposal = ActionProposal.model_validate(state['proposal'])
                self.lab.validate(proposal)
                action = self.lab.intent_action(proposal)
                if action['kind'] == 'finish':
                    # Native finish is a completion claim. Require the actual
                    # graph verification specialist before exposing that claim.
                    self.graph.update_state(self.config, {'route': 'verify', 'intent': None, 'proposal': None,
                        'candidates': [], 'operation_decision': None, 'operation_revision': None}, as_node='router')
                    value = None
                    continue
                self.pending = deepcopy(action)
                self.trace.emit('native_bridge', 'action_handoff', native_action=action, revision=self.lab.revision)
                return action
        except Exception as exc:
            if self.terminal_reason in (None, 'agent_supported'):
                self.terminal_reason = 'graph_policy_error'
            self.trace.emit('native_bridge', 'policy_abort', terminal_reason=self.terminal_reason,
                            error=f'{type(exc).__name__}: {exc}')
            raise GraphPolicyError(str(exc)) from exc
        finally:
            self.last_action_usage = self._usage(started)
