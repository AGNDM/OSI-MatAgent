"""Small operational prompts and read-only, per-capability native discovery.

Input schemas are copied without alteration. The native validator still sees
the original contracts. Simulation model narratives and provenance stay in the
saved public-contract.json instead of being repeated for every action.
"""
from copy import deepcopy
import json

from app.device_registry import public_capabilities
from app.material_retrieval import KnowledgeInstrumentProfile
from app.single_instrument import InstrumentLab, InstrumentProfile


DISCOVERY_INSTRUCTION = (
    ' The supplied capability menu contains exact input schemas and essential operating conditions. '
    'For output fields, error meanings or examples of one operation, propose workflow.describe '
    'with its capability_id. This calls the bound agent session.describe, costs one lab action, '
    'and neither advances time nor supplies a measurement or an inspection. '
    'Read observations.capability_detail after the query. Never call Python methods, host/evaluator '
    'configuration or unlisted capabilities. Examples are syntax references, not an action plan.'
)


def essential_preconditions(conditions):
    if isinstance(conditions, list):
        return [essential_preconditions(condition) for condition in conditions]
    if isinstance(conditions, dict):
        # Keep the full condition text, without suggesting evaluator state IDs
        # can be queried by the agent.
        result = {k: deepcopy(v) for k,v in conditions.items()
            if k in {'description_zh','description','enforced_by'}}
        public_ids = [key for key in conditions.get('state_ids', [])
            if not key.startswith(('truth.','host.','plant.','debug.'))]
        if public_ids:
            result['public_state_ids'] = public_ids
        return result
    return deepcopy(conditions)


def essential_capability(cap):
    result = {'input_schema': deepcopy(cap['input_schema'])}
    meaning = cap.get('function_zh') or cap.get('source') or cap.get('description')
    if meaning:
        result['description'] = deepcopy(meaning)
    if cap.get('preconditions'):
        result['preconditions'] = essential_preconditions(cap['preconditions'])
    execution = cap.get('execution') or cap.get('execution_notes')
    if isinstance(execution, dict):
        result['execution'] = {k: deepcopy(v) for k,v in execution.items() if k in {
            'mode','time_domain','completion_criterion','completion_state_ids',
            'timeout_s','cancellation','retry','partial_failure_zh'}}
    if cap.get('kind'):
        result['kind'] = cap['kind']
    return result


def essential_state(slot):
    return {k: deepcopy(v) for k,v in slot.items() if k in {
        'value_schema','schema','unit','meaning_zh','description','origin'}}


def lean_payload(payload):
    """Transform model input only; preserve the full source and readback values."""
    payload = deepcopy(payload)
    if 'capabilities' in payload:
        payload['capabilities'] = {name:essential_capability(cap)
            for name,cap in payload['capabilities'].items()}
    obs = payload.get('observations', {})
    contract = obs.get('instrument_contract')
    if contract:
        obs['instrument_contract'] = {
            'asset_type_id': contract.get('asset_type_id'),
            'state_slots': {key:essential_state(slot) for key,slot in contract.get('state_slots',{}).items()},
            'sampling': 'Readbacks retain native quality, sample_time_s and sample_window_s. Reads do not advance time.',
            'limitations': deepcopy(contract.get('limitations', []))}
    action = obs.get('last_action')
    if action:
        # Every intermediate boundary sample remains in trace/monitor files.
        # Planning needs the fresh measurement, error and elapsed-time receipt.
        result = action.get('result', {})
        for key in ('constraint_reads','clock_steps'):
            samples = result.pop(key, None)
            if samples is not None:
                result[key+'_count'] = len(samples)
        if result.get('polls'):
            result['last_poll'] = result['polls'][-1]
            result['poll_count'] = len(result.pop('polls'))
    return payload


class LeanInstrumentProfile(KnowledgeInstrumentProfile):
    prompts = {**KnowledgeInstrumentProfile.prompts,
        'diagnostic': KnowledgeInstrumentProfile.prompts['diagnostic'].replace(
            'only inspect/read/await/poll/cancel/stop', 'only inspect/read/describe/await/poll/cancel/stop'
        ) + DISCOVERY_INSTRUCTION}
    candidate_prompt = KnowledgeInstrumentProfile.candidate_prompt.replace(
        'only inspect/read/await/poll/cancel/stop', 'only inspect/read/describe/await/poll/cancel/stop'
    ) + DISCOVERY_INSTRUCTION

    @staticmethod
    def projection(role, state, request_id):
        # Bypass the old metadata-reference compactor: retain exact input
        # schemas, but do not serialize the removed non-operational metadata.
        return lean_payload(InstrumentProfile.projection(role, state, request_id))

    @staticmethod
    def model_payload(payload):
        # The complete search query is already emitted as knowledge_lookup.
        # Keep retrieved passages and citation IDs in the actual model input.
        payload = deepcopy(payload)
        payload.pop('retrieval_context', None)
        return payload


class DiscoverableInstrumentLab(InstrumentLab):
    """Opt-in wrapper; the ordinary instrument runner remains unchanged."""
    def __init__(self, *args, **kwargs):
        self.capability_detail = None
        super().__init__(*args, **kwargs)

    def capabilities(self):
        result = super().capabilities()
        result['workflow.describe'] = {
            'input_schema': {'type':'object', 'properties':{
                'capability_id':{'type':'string','enum':sorted(result)}},
                'required':['capability_id'],'additionalProperties':False},
            'source':'Read one public capability from the bound runtime description. No physical action, inspection or time advance.'}
        return result

    def observe(self):
        result = super().observe()
        if self.capability_detail:
            result['capability_detail'] = deepcopy(self.capability_detail)
        return result

    @staticmethod
    def action_cost(proposal):
        return 1 if proposal.capability_id == 'workflow.describe' else InstrumentLab.action_cost(proposal)

    def describe_capability(self, capability_id):
        # Only the bound, authenticated agent session is consulted. Caller
        # arguments cannot name a module, principal, path or private state.
        if capability_id.startswith('workflow.'):
            cap = super().capabilities()[capability_id]
            slots = self.description.get('state_slots', {})
            source = 'workflow wrapper contract'
        else:
            description = self.session.describe(instance_id=self.instance_id)
            self.lifecycle.calls['describe'] = self.lifecycle.calls.get('describe',0)+1
            if 'asset_types' in description:
                description = description['asset_types'][self.spec['asset_id']]
            native_id = capability_id.removeprefix(self.spec['slug']+'::')
            cap = public_capabilities(description)[native_id]
            slots = description.get('state_slots', {})
            source = 'bound agent session.describe'
        detail = essential_capability(cap)
        detail.update({k:deepcopy(cap[k]) for k in ('output_schema','example_arguments','errors','limitations') if k in cap})
        public_observes = [key for key in cap.get('observes',[]) if key in self.fields]
        if public_observes:
            detail['observes'] = {key:{**essential_state(slots[key]),
                'sampling':deepcopy(slots[key].get('sampling'))} for key in public_observes}
        return dict(capability_id=capability_id, source=source, contract=detail,
            simulation_time_s=self.lifecycle.elapsed, is_measurement=False)

    def execute(self, proposal):
        if proposal.capability_id != 'workflow.describe':
            return super().execute(proposal)
        self.validate(proposal)
        digest = json.dumps(proposal.model_dump(), sort_keys=True, allow_nan=False)
        if proposal.request_id in self._requests:
            return deepcopy(self._requests[proposal.request_id][1])
        self.action_count += 1
        try:
            self.capability_detail = self.describe_capability(proposal.arguments['capability_id'])
            result = {'capability_description':deepcopy(self.capability_detail)}
        except Exception as exc:
            self.termination_reason = 'capability_query_error'
            result = {'error':{'code':'CAPABILITY_QUERY_FAILED','detail':str(exc)[:1000]}}
        event = dict(step=self.action_count, proposal=proposal.model_dump(), revision=self.revision,
            result=result, pending_operation=deepcopy(self.pending_operation),
            source='Read-only capability query; no instrument invocation or clock advance')
        self.history.append(deepcopy(event))
        self._requests[proposal.request_id] = (digest,deepcopy(event))
        return event
