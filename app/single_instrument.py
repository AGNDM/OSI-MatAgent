"""Reference-free single-instrument execution over native public contracts.

No generated device task, reference probe or solution sequence is loaded here.
Clock/poll adapters follow native API signatures and contracts, not task scripts.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
import time
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import Field, field_validator

from app.models import Record
from app.device_registry import get_device, open_native, public_capabilities
from app.lab_adapter import GateRejected
from app.native_lifecycle import NativeLifecycle
from app.roles import projection
from app.device_roles import DeviceProfile


class StatePredicate(Record):
    state_id: str
    path: list[Any] = Field(default_factory=list, max_length=8)
    operator: Literal['eq', 'ge', 'le'] = 'eq'
    value: Any
    unit: str
    absolute_tolerance: float = Field(default=1e-6, ge=0)

    @field_validator('value')
    @classmethod
    def scalar(cls, value):
        if type(value) not in (str, bool, int, float, type(None)):
            raise ValueError('Predicate values must be finite scalars')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite predicate')
        return value

    @field_validator('path')
    @classmethod
    def path_parts(cls, parts):
        if any(type(p) not in (str, int) or (type(p) is int and p < 0) for p in parts):
            raise ValueError('Path parts must be string keys or nonnegative integer indices')
        return parts


class InstrumentTask(Record):
    task_id: str
    device: str
    objective: str
    predicates: list[StatePredicate] = Field(min_length=1)
    constraints: list[StatePredicate] = Field(default_factory=list)
    readable_state_ids: list[str] | None = None
    max_simulated_seconds: float = Field(default=3600, gt=0, le=86400)
    max_wait_seconds: float = Field(default=600, gt=0, le=3600)


def evaluate(predicate, native):
    """Evaluate only actual public readbacks, including native unit and quality."""
    try:
        if native.get('status') != 'succeeded':
            raise ValueError('read_unavailable')
        slot = native['observations'][predicate.state_id]
        if slot.get('quality') not in ('valid', 'estimated'):
            raise ValueError('quality_unavailable')
        if slot.get('unit') != predicate.unit:
            raise ValueError('unit_mismatch')
        clock = native.get('clock', {}).get('time_s')
        if clock is not None:
            timestamp = slot.get('sample_time_s')
            if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                    or abs(timestamp-clock) > 1e-6):
                raise ValueError('stale_or_unavailable_sample_time')
        value = slot['value']
        for key in predicate.path:
            if type(key) is str and isinstance(value, dict):
                value = value[key]
            elif type(key) is int and isinstance(value, list):
                value = value[key]
            else:
                raise ValueError('path_type_mismatch')
        expected = predicate.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError('non_finite_value')
        numeric = type(value) in (float, int) and type(expected) in (float, int)
        if predicate.operator != 'eq' and not numeric:
            raise ValueError('numeric_comparison_type_mismatch')
        if numeric:
            tolerance = predicate.absolute_tolerance
            ok = (abs(value-expected) <= tolerance if predicate.operator == 'eq' else
                  value >= expected-tolerance if predicate.operator == 'ge' else value <= expected+tolerance)
        else:
            ok = type(value) is type(expected) and value == expected
        return {'state_id': predicate.state_id, 'path': predicate.path, 'met': bool(ok),
                'actual': value, 'expected': expected, 'error': None}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return {'state_id': predicate.state_id, 'path': predicate.path, 'met': False,
                'error': str(exc)}


class InstrumentLab:
    manages_pending_operations = True
    count_router_calls = True
    router_wait_enabled = True

    def __init__(self, task: InstrumentTask, *, native=None):
        self.task = task
        self.spec = get_device(task.device)
        # Optional native injection is for offline contract tests only, never CLI input.
        self.host, self.session, self.instance_id, self.description = native or open_native(self.spec)
        self.lifecycle = NativeLifecycle(self.spec, self.host, self.session, self.instance_id, self.description)
        self.sample_id = 'instrument:' + self.spec['slug']
        self.revision = self.action_count = self.last_mutation_step = 0
        self.pending_operation = None
        self.termination_reason = None
        self._inspection = self._measurement = None
        self.history, self._requests, self.protocol_errors = [], {}, []
        slots = self.description.get('state_slots', {})
        public = {k: v for k, v in slots.items() if v.get('visibility', 'agent') == 'agent'
                  and not k.startswith(('truth.', 'debug.', 'host.', 'plant.'))}
        self.fields = list(task.readable_state_ids) if task.readable_state_ids is not None else list(public)
        if not self.fields or set(self.fields)-public.keys():
            raise ValueError('Readable states must be declared public states')
        for p in task.predicates + task.constraints:
            if p.state_id not in self.fields or public[p.state_id].get('unit') != p.unit:
                raise ValueError('Predicate references an unreadable state or invalid unit')
        self._native_caps = public_capabilities(self.description)
        self._last_native_revision = -1
        self._initial = self._read()

    def _read(self):
        self.lifecycle.calls['read'] += 1
        response = self.session.read(instance_id=self.instance_id, state_ids=self.fields)
        self._validate_receipt(response, read=True)
        return response

    def _validate_receipt(self, response, *, read=False):
        if response.get('instance_id') != self.instance_id:
            raise ValueError('NATIVE_INSTANCE_MISMATCH')
        clock = response.get('clock', {}).get('time_s')
        if type(clock) not in (int, float) or not math.isfinite(clock):
            raise ValueError('NATIVE_CLOCK_MISSING')
        expected = self._initial_clock + self.lifecycle.elapsed if hasattr(self, '_initial_clock') else clock
        if abs(clock - expected) > 1e-6:
            raise ValueError('NATIVE_CLOCK_MISMATCH')
        if not hasattr(self, '_initial_clock'):
            self._initial_clock = clock
            self.lifecycle.initial_clock = clock
        revision = response.get('state_revision')
        if type(revision) is not int or revision < self._last_native_revision:
            raise ValueError('NATIVE_REVISION_MISMATCH')
        self._last_native_revision = revision

    def target(self):
        return {'task_id': self.task.task_id, 'sample_id': self.sample_id,
                'objective': self.task.objective,
                'predicates': [p.model_dump() for p in self.task.predicates],
                'constraints': [p.model_dump() for p in self.task.constraints],
                'limits': {'max_simulated_seconds': self.task.max_simulated_seconds,
                           'max_wait_seconds': self.task.max_wait_seconds},
                'scope': 'Public native-state goals; no reference sequence or scientific outcome claim.'}

    def capabilities(self):
        def cap(properties, source):
            return {'input_schema': {'type': 'object', 'properties': properties,
                    'required': list(properties), 'additionalProperties': False}, 'source': source}
        result = {'workflow.inspect': cap({}, 'Read the public native state and instrument contract.'),
                  'workflow.readback': cap({}, 'Read fresh public state; does not advance simulation time.'),
                  'workflow.stop': cap({'reason': {'type': 'string', 'minLength': 1, 'maxLength': 500}},
                                       'Stop honestly as unresolved; does not cancel a pending operation.')}
        if self.lifecycle.can_step:
            result['workflow.await'] = cap({'duration_s': {'type': 'number', 'exclusiveMinimum': 0,
                  'maximum': self.task.max_wait_seconds}}, 'Host advances native clock in bounded chunks, polls supported pending operations, then reads fresh public state.')
        if self.lifecycle.can_poll:
            result['workflow.poll'] = cap({}, 'Poll the current operation without advancing time.')
        if self.lifecycle.can_cancel:
            result['workflow.cancel'] = cap({}, 'Cancel the current operation; partial native effects are preserved.')
        for name, metadata in self._native_caps.items():
            result[self.spec['slug']+'::'+name] = deepcopy(metadata)
        return result

    def observe(self):
        return deepcopy({'revision': self.revision, 'sample_id': self.sample_id,
            'inspection': self._inspection, 'measurement': self._measurement,
            'last_mutation_step': self.last_mutation_step, 'pending_operation': self.pending_operation,
            'last_action': self.history[-1] if self.history else None,
            'simulated_seconds': self.lifecycle.elapsed,
            'instrument_contract': {'asset_type_id': self.description.get('asset_type_id'),
                'state_slots': {k: self.description['state_slots'][k] for k in self.fields},
                'limitations': self.description.get('limitations', [])}})

    def _snapshot(self):
        native = self._read()
        result = {'id': f'obs-{self.action_count}', 'sample_id': self.sample_id,
                  'revision': self.revision, 'captured_at_step': self.action_count,
                  'quality': 'valid' if native.get('status') == 'succeeded' else 'unavailable', 'native': native}
        if any(not evaluate(p, native)['met'] for p in self.task.constraints):
            self.termination_reason = 'constraint_violated'
        return result

    def _invalidate(self):
        self.revision += 1
        self.last_mutation_step = self.action_count
        self._measurement = None

    def _refresh(self):
        self.action_count += 1
        self._measurement = self._snapshot()
        self._inspection = deepcopy(self._measurement)

    @staticmethod
    def action_cost(proposal):
        return 1 if proposal.capability_id in {'workflow.inspect', 'workflow.readback', 'workflow.stop'} else 2

    def validate(self, proposal, pending_operation=None):
        digest = json.dumps(proposal.model_dump(), sort_keys=True, allow_nan=False)
        if proposal.sample_id != self.sample_id:
            raise GateRejected('WRONG_DEVICE')
        if proposal.request_id in self._requests:
            if self._requests[proposal.request_id][0] != digest:
                raise GateRejected('REQUEST_CONFLICT')
            return
        if self.termination_reason:
            raise GateRejected('EPISODE_TERMINATED')
        if proposal.expected_revision != self.revision:
            raise GateRejected('STALE_PROPOSAL')
        cap = self.capabilities().get(proposal.capability_id)
        if cap is None or list(Draft202012Validator(cap['input_schema']).iter_errors(proposal.arguments)):
            raise GateRejected('UNSUPPORTED_CAPABILITY_OR_INVALID_ARGUMENT')
        if self.pending_operation and proposal.capability_id not in {
                'workflow.inspect', 'workflow.readback', 'workflow.describe', 'workflow.await', 'workflow.poll', 'workflow.cancel', 'workflow.stop'}:
            raise GateRejected('PENDING_OPERATION_UNRESOLVED')
        if proposal.capability_id in {'workflow.poll', 'workflow.cancel'} and not self.pending_operation:
            raise GateRejected('NO_PENDING_OPERATION')
        if proposal.capability_id == 'workflow.await' and self.lifecycle.elapsed + proposal.arguments['duration_s'] > self.task.max_simulated_seconds:
            raise GateRejected('SIMULATION_BUDGET')
        if (getattr(self, 'require_initial_inspection', True)
                and not proposal.capability_id.startswith('workflow.') and not self._inspection):
            raise GateRejected('INSPECTION_REQUIRED')

    def _update_pending(self, response, capability):
        self._validate_receipt(response)
        status = self.lifecycle.operation_status(response)
        response_id = response.get('operation_id') or response.get('result', {}).get('operation_id')
        if self.pending_operation and response_id and response_id != self.pending_operation['operation_id']:
            self.termination_reason = 'operation_identity_mismatch'
            self.protocol_errors.append('Operation identity changed in native response')
            return status
        if status in {'accepted', 'pending', 'running'}:
            ident = response.get('operation_id') or response.get('result', {}).get('operation_id')
            if self.pending_operation:
                previous_id = self.pending_operation['operation_id']
                if ident and ident != previous_id:
                    self.termination_reason = 'operation_identity_mismatch'
                    self.protocol_errors.append('Operation identity changed in native response')
                    return status
                ident = previous_id
            if not ident:
                self.termination_reason = 'pending_missing_operation_id'
                self.pending_operation = {'operation_id': None, 'capability': capability, 'status': status}
            else:
                self.pending_operation = {'operation_id': ident, 'capability': capability, 'status': status}
                if not self.lifecycle.can_poll:
                    self.termination_reason = 'unsupported_pending_lifecycle'
        elif self.pending_operation:
            if status in {'succeeded', 'failed', 'cancelled', 'canceled'} and response_id:
                self.pending_operation = None
            else:
                # Poll rejection does not establish what happened to the accepted op.
                self.termination_reason = 'operation_status_uncertain'
        return status

    def execute(self, proposal):
        self.validate(proposal)
        digest = json.dumps(proposal.model_dump(), sort_keys=True, allow_nan=False)
        if proposal.request_id in self._requests:
            return deepcopy(self._requests[proposal.request_id][1])
        self.action_count += 1
        name = proposal.capability_id
        result = {}
        try:
            if name in {'workflow.inspect', 'workflow.readback'}:
                self._measurement = self._snapshot()
                self._inspection = deepcopy(self._measurement)
                result['measurement'] = deepcopy(self._measurement)
            elif name == 'workflow.stop':
                self.termination_reason = 'agent_stopped_unresolved'
                result = {'stopped': True, 'reason': proposal.arguments['reason'],
                          'pending_operation_retained': deepcopy(self.pending_operation)}
            else:
                # Conservative even for rejected calls: partial/uncertain effects cannot reuse old evidence.
                self._invalidate()
                if name == 'workflow.await':
                    left = proposal.arguments['duration_s'];polls=[];steps=[];constraint_reads=[]
                    while left > 1e-9:
                        deadline = getattr(self, 'deadline', float('inf'))
                        if time.monotonic() >= deadline:
                            self.termination_reason = 'deadline'
                            break
                        dt = min(getattr(self, 'await_step_seconds', 10.), left)
                        self.lifecycle.step(dt);left -= dt
                        steps.append({'dt_s':dt, 'elapsed_s':self.lifecycle.elapsed})
                        if self.task.constraints:
                            observed = self._read()
                            constraint_reads.append(observed)
                            if any(not evaluate(p, observed)['met'] for p in self.task.constraints):
                                self.termination_reason = 'constraint_violated'
                                result['constraint_observation'] = observed
                                break
                        if self.termination_reason:
                            break
                        if self.pending_operation and getattr(self, 'auto_poll_during_wait', True):
                            response = self.lifecycle.poll(self.pending_operation['operation_id'])
                            polls.append(response)
                            status = self._update_pending(response, self.pending_operation['capability'])
                            if self.termination_reason or not self.pending_operation:
                                if status in {'failed', 'cancelled', 'canceled'}:
                                    result['error'] = response.get('error') or {'code': 'OPERATION_'+status.upper()}
                                break
                    result.update(advanced_seconds=proposal.arguments['duration_s']-left, polls=polls, clock_steps=steps, constraint_reads=constraint_reads)
                elif name in {'workflow.poll', 'workflow.cancel'}:
                    pending = deepcopy(self.pending_operation)
                    response = (self.lifecycle.poll(pending['operation_id']) if name == 'workflow.poll' else
                                self.lifecycle.cancel(pending['operation_id'], proposal.request_id))
                    result['native_response'] = response
                    status = self._update_pending(response, pending['capability'])
                    if status not in {'succeeded', 'running', 'pending', 'accepted'}:
                        result['error'] = response.get('error') or {'code': 'OPERATION_'+str(status).upper()}
                else:
                    native_name = name.split('::', 1)[1]
                    self.lifecycle.calls['invoke'] += 1
                    response = self.session.invoke(instance_id=self.instance_id, capability_id=native_name,
                                  arguments=proposal.arguments, request_id=proposal.request_id)
                    result['native_response'] = response
                    status = self._update_pending(response, native_name)
                    if status not in {'succeeded', 'accepted', 'pending', 'running'}:
                        result['error'] = response.get('error') or {'code': 'NATIVE_'+str(status)}
                self._refresh()
        except Exception as exc:
            self._measurement = None
            self.termination_reason = 'execution_uncertain'
            self.protocol_errors.append('Unconfirmed native/clock effects: '+type(exc).__name__)
            result['error'] = {'code': 'EXECUTION_UNCERTAIN', 'detail': str(exc)[:1000]}
        event = {'step': self.action_count, 'proposal': proposal.model_dump(), 'revision': self.revision,
                 'result': result, 'pending_operation': deepcopy(self.pending_operation), 'source': 'native public session; no task action script'}
        self.history.append(deepcopy(event))
        self._requests[proposal.request_id] = (digest, deepcopy(event))
        return event

    def score(self, report):
        final = self._read()
        predicates = [evaluate(p, final) for p in self.task.predicates]
        constraints = [evaluate(p, final) for p in self.task.constraints]
        m = self._measurement
        fresh = bool(m and m['quality'] == 'valid' and m['revision'] == self.revision and
                     m['captured_at_step'] > self.last_mutation_step)
        if fresh:
            fresh = all(evaluate(p, m['native']).get('error') is None
                        for p in self.task.predicates + self.task.constraints)
        # A final independent read may differ (e.g. noisy readout); report both, never hide it.
        cached = bool(m and all(evaluate(p, m['native'])['met'] for p in self.task.predicates))
        met = bool(all(p['met'] for p in predicates) and all(p['met'] for p in constraints) and cached and fresh
                   and not self.pending_operation and self.termination_reason not in {'constraint_violated','execution_uncertain'})
        claim = report.get('status') == 'supported'
        cited = bool(m and m['id'] in report.get('evidence_ids', []))
        return {'goal_met': met, 'task_success': met, 'agent_claim_supported': claim,
            'false_success': claim and not (met and cited), 'protocol_correct': not self.protocol_errors,
            'protocol_errors': deepcopy(self.protocol_errors), 'fresh_evidence': fresh, 'evidence_cited': cited,
            'predicates': predicates, 'constraints': constraints, 'pending_operation': deepcopy(self.pending_operation),
            'lifecycle_resolved': self.pending_operation is None and not self.protocol_errors,
            'simulated_seconds': self.lifecycle.elapsed, 'native_io_counts': deepcopy(self.lifecycle.calls),
            'terminal_reason': self.termination_reason or report.get('terminal_reason')}

    def operation_context(self, context, candidates):
        # Chooser sees public state and supplied options, not a repeated full device manual.
        context = deepcopy(context)
        context.pop('capabilities', None)
        obs = context.get('observations') or {}
        obs.pop('instrument_contract', None)
        action = obs.get('last_action')
        if action:
            obs['last_action'] = {'proposal': action.get('proposal'), 'error': action.get('result', {}).get('error')}
        for key in ['inspection', 'measurement']:
            snapshot = obs.get(key)
            if snapshot and 'native' in snapshot:
                snapshot['native'] = {'observations': {k: {field: v.get(field) for field in ['value','unit','quality']}
                    for k,v in snapshot['native'].get('observations', {}).items()}}
        return context

    def active_goal_process(self, native):
        """Conservative public lifecycle guard, scoped to goal state families.

        Unknown lifecycle semantics use the ordinary expert path. This guard
        neither chooses an action sequence nor assumes command success is progress.
        """
        if self.pending_operation:
            return True
        families = {p.state_id.split('.')[0] for p in self.task.predicates}
        active = False
        for state_id, slot in native.get('observations', {}).items():
            if state_id.split('.')[0] not in families or slot.get('quality') not in {'valid', 'estimated'}:
                continue
            value = slot.get('value')
            markers = value if isinstance(value, dict) else {state_id.rsplit('.', 1)[-1]: value}
            for key, marker in markers.items():
                if key in {'phase', 'status', 'state'} and isinstance(marker, str):
                    if marker.lower() in {'paused', 'blocked', 'fault', 'failed', 'stopped', 'idle'}:
                        return False
                    active |= marker.lower() in {'running', 'ramp', 'hold', 'ramping', 'holding'}
                if key == 'running':
                    if marker is False:
                        return False
                    active |= marker is True
        return active

    def router_wait_offers(self, state):
        obs = state['observations']
        snapshot = obs.get('measurement') or obs.get('inspection')
        fresh = bool(snapshot and snapshot['quality'] == 'valid' and snapshot['revision'] == self.revision
                     and snapshot['captured_at_step'] > self.last_mutation_step)
        if (not self.router_wait_enabled or not self.lifecycle.can_step or not fresh
                or state.get('last_error') or self.termination_reason):
            return {}
        native = snapshot['native']
        if not self.active_goal_process(native):
            return {}
        if (all(evaluate(p,native)['met'] for p in self.task.predicates)
                or not all(evaluate(p,native)['met'] for p in self.task.constraints)):
            return {}
        remaining = self.task.max_simulated_seconds-self.lifecycle.elapsed
        maximum = min(self.task.max_wait_seconds, remaining)
        if maximum <= 1e-9:
            return {}
        short, long = min(30., maximum), min(120., maximum)
        return {'wait_short': short, **({'wait_long': long} if long > short else {})}

    def verification_signature(self, state):
        """Identify substantive public evidence, independent of read/observation IDs.

        Wrapper revision/capture counters alone can change on a no-op read or
        rejected write. Native values, timestamps, lifecycle and quality determine
        whether another assessment can supply new information.
        """
        obs = state['observations']
        snapshot = obs.get('measurement')
        if not snapshot:
            return None
        native = snapshot.get('native') or {}
        relevant = {'task': self.target(), 'sample_id': snapshot.get('sample_id'),
            'quality': snapshot.get('quality'),
            'fresh_revision': snapshot.get('revision') == obs.get('revision'),
            'fresh_capture': snapshot.get('captured_at_step', -1) > obs.get('last_mutation_step', 0),
            'pending_operation': obs.get('pending_operation'),
            'native': {key: native.get(key) for key in
                ['instance_id', 'status', 'clock', 'observations']}}
        return sha256(json.dumps(relevant,sort_keys=True,ensure_ascii=False,
                                 allow_nan=False).encode()).hexdigest()

    def route_context(self, state, summary, allowed):
        if self.pending_operation:
            summary = dict(summary, has_measurement=False)
            allowed = [r for r in allowed if r != 'verify']
        if not self.router_wait_enabled:
            return summary, allowed
        # At most four one-token choices. Ordinary stage transfers remain model-selected.
        allowed = [r for r in allowed if r != 'deliberate']
        offers = self.router_wait_offers(state) if not state.get('intent') else {}
        allowed += list(offers)
        obs = state['observations']
        snapshot = obs.get('measurement') or obs.get('inspection')
        native = snapshot['native'] if snapshot else {}
        signature = self.verification_signature(state)
        already_verified = bool(signature and signature in state.get('verified_evidence_signatures', []))
        recent = [{'capability_id': e['proposal']['capability_id'], 'arguments': e['proposal']['arguments'],
                   'error': e['result'].get('error'), 'advanced_seconds': e['result'].get('advanced_seconds'),
                   'native_status': e['result'].get('native_response',{}).get('status')}
                  for e in self.history[-4:]]
        intent = deepcopy(state.get('intent'))
        intent_current = bool(intent and state.get('observation_revision') == self.revision
            and (not state.get('operation_decision') or state.get('operation_revision') == self.revision))
        context = {'selected_intent': intent, 'selected_intent_current': intent_current,
            'objective': self.task.objective, 'predicates': [p.model_dump() for p in self.task.predicates],
            'goal_checks': [evaluate(p,native) for p in self.task.predicates],
            'constraint_checks': [evaluate(p,native) for p in self.task.constraints],
            'public_state': {k: {field:v.get(field) for field in ['value','unit','quality','sample_time_s']}
                             for k,v in native.get('observations',{}).items()},
            'observation_id': snapshot['id'] if snapshot else None,
            'revision': self.revision, 'native_clock': native.get('clock'),
            'pending_operation': deepcopy(self.pending_operation), 'recent_actions': recent,
            'last_error': state.get('last_error'), 'elapsed_simulated_seconds': self.lifecycle.elapsed,
            'remaining_simulated_seconds': self.task.max_simulated_seconds-self.lifecycle.elapsed,
            'wait_offers': offers, 'last_verification': deepcopy(state.get('last_verification_feedback')),
            'current_evidence_already_verified': already_verified}
        if already_verified:
            allowed = [route for route in allowed if route != 'verify']
        if summary.get('has_intent'):
            summary = dict(summary, stage='prepare')
        elif already_verified or not all(evaluate(p,native)['met'] for p in self.task.predicates):
            summary = dict(summary, stage='plan_or_wait')
        return dict(summary, instrument_context=context), allowed


class InstrumentProfile(DeviceProfile):
    prompts = {
        'diagnostic': 'Solve the stated single-instrument objective using native public contracts and observations. No reference action sequence is supplied. Inspect first. Use input_schema, optional fields, units, preconditions and actual errors to plan one next operation. A succeeded command may only start a continuing physical process. To let supported native dynamics progress request workflow.await with a bounded duration_s, then use fresh readback. Reads/polls never advance time. While pending, only inspect/read/await/poll/cancel/stop; do not start another native action. If prerequisites require unavailable host fixtures, or the goal is unsupported, use workflow.stop with an honest reason rather than inventing host calls. Cite only supplied inspection/measurement observation IDs; documentation IDs such as native-contract and state IDs are not observations. With no observations cite empty lists. Return short hypotheses and one intent; never execute code.',
        'procedure': 'Copy the supplied intent into exactly one ActionProposal. Copy sample_id, expected_revision, request_id, capability_id and arguments exactly. Do not invent steps, optional arguments or native operations.',
        'verification': 'Verify the declared public-state predicates and constraints using fresh measurement.native.observations and unit/quality/path fields. Native qualities valid AND estimated are both accepted by this simulator task policy; goal evidence also needs a current finite sample timestamp matching the native read clock; estimated does not mean missing evidence. All predicates and constraints must hold; numeric eq uses absolute_tolerance. Evidence must match sample_id and revision and be captured after last_mutation_step. Never report supported while pending_operation exists. Cite the measurement id. If fresh accepted-quality evidence shows ANY evaluable predicate false, report contradicted; do not describe an observed mismatch as missing evidence. Report unresolved only for pending operations, stale/missing/unavailable/invalid identity, units, paths or evidence quality. Report supported only when every predicate and constraint holds. Do not confuse command acceptance with physical completion.',
        'deliberation': 'Select one allowed specialist route. Prepare an existing intent; pending operations require diagnosis of await/poll/stop; verify only fresh completed evidence. Never execute tools.'}
    candidate_prompt = (prompts['diagnostic'] + ' Keep output compact: at most one short hypothesis and two candidates. Each prose field must be under 120 characters. Never reproduce contracts in the output. At the initial state without inspection return one workflow.inspect candidate with empty arguments and empty evidence IDs only.')

    @staticmethod
    def projection(role, state, request_id):
        payload = projection(role, state, request_id)
        if role == 'diagnostic':
            payload['last_verification'] = deepcopy(state.get('last_verification_feedback'))
        elif role == 'procedure':
            name = payload['intent']['capability_id']
            payload['capabilities'] = {name: payload['capabilities'][name]}
        elif role == 'verification':
            payload = deepcopy(payload)
            payload['observations'].pop('instrument_contract', None)
            payload['observations'].pop('last_action', None)
            payload['sample_id'] = state['sample_id']
            payload['verification_policy'] = {'accepted_native_qualities':['valid','estimated'],
                'mismatch':'contradicted','unavailable_or_pending':'unresolved',
                'all_predicates_and_constraints_hold':'supported'}
        return payload

    @staticmethod
    def retrieve(role):
        return [{'id': 'native-contract', 'source': 'authenticated native session.describe',
                 'text': 'Instrument capability schemas, optional arguments, preconditions and public state descriptions are supplied directly. No task reference script is used.'}]

    @staticmethod
    def validate_role_contract(role, record, payload):
        DeviceProfile.validate_role_contract(role, record, payload)
        if role == 'verification':
            expected = verification_status(payload)
            if record.status != expected:
                raise ValueError('Verification evidence contract requires '+expected+
                    '; valid and estimated are accepted, evaluable mismatches are not missing evidence')
            if expected in {'supported', 'contradicted'}:
                measurement = payload['observations']['measurement']
                if measurement['id'] not in record.evidence_ids:
                    raise ValueError('Cite the current measurement ID for an evaluable verification verdict')


def verification_status(payload):
    """Shared native predicate evaluator; validation never rewrites model verdicts."""
    obs = payload['observations']
    m = obs.get('measurement')
    if (not m or obs.get('pending_operation') or m.get('quality') != 'valid'
            or m.get('sample_id') != (payload.get('sample_id') or payload['target'].get('sample_id'))
            or m.get('revision') != obs.get('revision')
            or m.get('captured_at_step', -1) <= obs.get('last_mutation_step', 0)):
        return 'unresolved'
    checks = [evaluate(StatePredicate.model_validate(p), m.get('native', {}))
        for p in payload['target']['predicates'] + payload['target'].get('constraints', [])]
    # A known false condition establishes conjunction failure even when another
    # field is unavailable. Missing evidence alone cannot establish success.
    if any(not c['met'] and c.get('error') is None for c in checks):
        return 'contradicted'
    if any(c.get('error') is not None for c in checks):
        return 'unresolved'
    return 'supported'
