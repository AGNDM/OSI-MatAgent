"""Opt-in agent-directed controls and observations; monitor data stays private."""
from copy import deepcopy

from app.action_memory import attach_public_memory, MEMORY_INSTRUCTION
from app.device_roles import DeviceProfile
from app.models import INTENT_OUTPUT_INSTRUCTION
from app.lean_instrument import DiscoverableInstrumentLab, essential_capability, lean_payload
from app.single_instrument import InstrumentProfile
from app.router_context import selected_intent_context


DIAGNOSTIC = (
    'Work toward the objective using the listed public operations. Choose controls, observation fields, '
    'and observation timing yourself; no inspection-first or read-after-write sequence is required. '
    'Use the exact input schemas and units. workflow.describe provides details of one operation. '
    'Only workflow.await advances simulation time; controls and reads do not. Controls and waits '
    'do not automatically supply sensor readings. Historical observations keep their original timestamps. '
    'The independent evaluator stops on completion or a boundary; its monitoring samples and chemical '
    'composition are not available as agent observations. Do not infer measured conversion from uncalibrated heat flow. '
    'Use workflow.stop if you cannot proceed. Cite only supplied observation IDs, with empty lists when '
    'none exist. Keep prose brief; do not execute code.'
) + ' ' + MEMORY_INSTRUCTION


class AutonomousInstrumentProfile(InstrumentProfile):
    prompts = {**InstrumentProfile.prompts, 'diagnostic': DIAGNOSTIC}
    candidate_prompt = DIAGNOSTIC
    candidate_instruction = (
        ' Return OperationOptions with hypotheses and 1-4 distinct candidate next operations, '
        'each with candidate_id, intent, rationale and evidence_ids. A separate decision model '
        'selects one. Do not rank candidates or mark a preferred one. No operation type is mandatory.'
    ) + INTENT_OUTPUT_INSTRUCTION

    @staticmethod
    def projection(role, state, request_id):
        payload = lean_payload(InstrumentProfile.projection(role, state, request_id))
        if "observations" not in payload:
            for key in ("operation_memory", "snapshot_status"):
                payload[key] = deepcopy(state["observations"].get(key))
        return payload

    @staticmethod
    def model_payload(payload):
        payload = deepcopy(payload)
        payload.pop('retrieval_context', None)
        if 'operation_context' in payload:
            obs = payload['observations']
            snapshot = obs.get('measurement') or obs.get('inspection')
            payload['operation_context'] = {
                'last_observation_id': snapshot.get('id') if snapshot else None,
                'last_observation_clock': snapshot.get('native', {}).get('clock') if snapshot else None,
                'observation_revision': snapshot.get('revision') if snapshot else None,
                'current_revision': obs['revision'],
                'instruction': 'Choose whether a new observation is needed. No automatic sensor readback.'}
        return payload

    @staticmethod
    def retrieve(role):
        return []

    validate_role_contract = staticmethod(DeviceProfile.validate_role_contract)


class AutonomousInstrumentLab(DiscoverableInstrumentLab):
    require_initial_inspection = False
    auto_poll_during_wait = False
    router_wait_enabled = False

    def __init__(self, *args, **kwargs):
        self._requested_state_ids = None
        super().__init__(*args, **kwargs)

    def target(self):
        # Internal evaluator predicates are not a recipe for the controller.
        return {'task_id': self.task.task_id, 'sample_id': self.sample_id,
            'objective': self.task.objective,
            'limits': {'max_simulated_seconds': self.task.max_simulated_seconds,
                       'max_wait_seconds': self.task.max_wait_seconds},
            'scope': 'Fixed initial batch; no material addition/discharge or kinetic-model changes. '
                     'Completion and boundaries are checked independently; monitor data is private.'}

    def capabilities(self):
        result = super().capabilities()
        for name in ('workflow.inspect', 'workflow.readback'):
            result[name] = {'input_schema': {'type': 'object', 'properties': {
                'state_ids': {'type': 'array', 'items': {'type': 'string', 'enum': sorted(self.fields)},
                              'minItems': 1, 'uniqueItems': True}},
                'required': [], 'additionalProperties': False},
                'source': 'Read chosen public state_ids (omit to read all). No time advance.'}
        if 'workflow.await' in result:
            result['workflow.await']['source'] = (
                'Advance simulation by duration_s, with independent boundary/completion checks. '
                'No automatic sensor readback or operation-status poll; may stop before duration_s.')
        return result

    def describe_capability(self, capability_id):
        if capability_id.startswith('workflow.'):
            return dict(capability_id=capability_id, source='workflow wrapper contract',
                contract=essential_capability(self.capabilities()[capability_id]),
                simulation_time_s=self.lifecycle.elapsed, is_measurement=False)
        return super().describe_capability(capability_id)

    @staticmethod
    def action_cost(proposal):
        return 1

    def _snapshot(self):
        ids = self.fields if self._requested_state_ids is None else self._requested_state_ids
        self.lifecycle.calls['read'] += 1
        self.lifecycle.calls['agent_read'] = self.lifecycle.calls.get('agent_read', 0) + 1
        native = self.session.read(instance_id=self.instance_id, state_ids=ids)
        self._validate_receipt(native, read=True)
        # Independent checks never become part of the requested observation.
        self._read()
        return {'id': f'obs-{self.action_count}', 'sample_id': self.sample_id,
                'revision': self.revision, 'captured_at_step': self.action_count,
                'quality': 'valid' if native.get('status') == 'succeeded' else 'unavailable',
                'native': native}

    def _refresh(self):
        # Called after a control/wait by InstrumentLab: monitor only, no agent read.
        self._read()

    def execute(self, proposal):
        self._requested_state_ids = proposal.arguments.get('state_ids') if proposal.capability_id in {
            'workflow.inspect', 'workflow.readback'} else None
        try:
            return super().execute(proposal)
        finally:
            self._requested_state_ids = None

    def observe(self):
        result = super().observe()
        # Raw execution/monitor logs retain these for audit; no model sees them.
        action = result.get('last_action')
        if action:
            data = action.get('result', {})
            for key in ('constraint_reads', 'clock_steps'):
                rows = data.pop(key, None)
                if rows is not None:
                    data[key + '_count'] = len(rows)
            data.pop('constraint_observation', None)
        return attach_public_memory(result, self.history)

    def route_context(self, state, summary, allowed):
        summary, allowed = super().route_context(state, summary, allowed)
        summary = deepcopy(summary)
        for key in ('operation_memory', 'snapshot_status'):
            summary[key] = deepcopy(state['observations'].get(key))
        summary.update(selected_intent_context(state, self.revision))
        return summary, allowed

    def operation_context(self, context, candidates):
        result = super().operation_context(context, candidates)
        # Retain the public last receipt too; the base chooser projection drops it.
        original_action = context.get('observations', {}).get('last_action') or {}
        native_receipt = (original_action.get('result') or {}).get('native_response')
        if native_receipt and result['observations'].get('last_action'):
            memory = context['observations'].get('operation_memory') or {}
            rows = memory.get('recent_actions') or []
            if rows and rows[-1].get('receipt'):
                result['observations']['last_action']['receipt_ref'] = 'operation_memory.recent_actions[-1].receipt'
        # Preserve age/identity of voluntarily requested observations for Jev too.
        for key in ('inspection', 'measurement'):
            original = context.get('observations', {}).get(key)
            trimmed = result.get('observations', {}).get(key)
            if original and trimmed:
                trimmed['native']['clock'] = deepcopy(original['native'].get('clock'))
                for name, slot in trimmed['native']['observations'].items():
                    slot['sample_time_s'] = original['native']['observations'][name].get('sample_time_s')
                    if 'sample_window_s' in original['native']['observations'][name]:
                        slot['sample_window_s'] = deepcopy(original['native']['observations'][name]['sample_window_s'])
        compared = {candidate['intent']['capability_id'] for candidate in candidates}
        result['candidate_capabilities'] = {key: deepcopy(cap)
            for key, cap in context.get('capabilities', {}).items() if key in compared}
        return result


DEVICE_CHOICES = ('ta_dsc2500', 'ika_c6000', 'mt_rc1mxhfcal', 'ta_tamiv')
SELECTION_PROMPT = (
    'Choose one instrument for the objective from the supplied directory. '
    'Use describe to obtain a brief public capability summary before proposing a device. '
    'Propose 1-4 distinct candidates with brief rationales; a separate decision model selects one. '
    'No device is preassigned. Only one device can be used; sample transfer is unavailable. '
    'Use stop if no option is suitable. Return DiscoveryRequest JSON within the query budget.'
)


def brief_device(catalog, device_id):
    from app.device_catalog import describe
    item = describe(catalog, device_id)
    return {**{key: item[key] for key in ('id', 'name', 'domain', 'lifecycle')},
            'capability_ids': sorted(item['capabilities']),
            'state_units': {key: slot['unit'] for key, slot in item['state_slots'].items()}}


def selection_options(task):
    from app.device_catalog import load_catalog
    catalog = load_catalog()
    catalog['devices'] = [next(d for d in catalog['devices'] if d['id'] == key) for key in DEVICE_CHOICES]
    return {'catalog': catalog, 'public_task': {'objective': task.objective},
            'prompt': SELECTION_PROMPT, 'describe_device': brief_device}
