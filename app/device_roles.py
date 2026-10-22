"""Device-specific prompts on the existing diagnose/procedure/gate/execute graph."""
from copy import deepcopy

from app.models import ActionProposal, HypothesisSet, RouteDecision, VerificationReport
from app.roles import projection
from app.graph import validate_role_contract
from app.device_lab import matches

PROMPTS = {
    'diagnostic': 'Operate the named virtual instrument using only the supplied capability menu. First inspect if inspection is absent. Compare the public state with target.goal, select a capability and exact schema-valid arguments to reach that goal. If a native operation failed, inspect last_error and choose a justified next step. Successful native tools automatically return a fresh readback as observations.measurement. After a failed command inspect last_error and choose a repair; do not assume success. If no post-action measurement is available, request workflow.readback before another write. An inspection from an older revision is stale; last_action is the actual native receipt, so do not repeat a command because the old inspection is unchanged. Request workflow.readback to obtain fresh verification evidence. Never repeatedly write a goal already established. Cite observation IDs, not state IDs. Native examples explain parameter spelling and units; they are not evidence of task completion. Return one short hypothesis (a cause label and evidence IDs) and one intent with brief strings, never execute tools. Retrieved contract IDs are documentation, not observation IDs; never cite them as measurements.',
    'procedure': 'Translate the supplied intent to one action proposal. Copy sample_id, expected_revision, request_id, capability_id and arguments exactly. The sample_id field is a device scope identity here. Do not execute tools or invent arguments.',
    'verification': 'Verify only the stated device-control goal, not chemical completion. Use observations.measurement.native.observations[target.goal.state_id]. Its quality must be valid or estimated, value must equal target.goal.equals (numeric tolerance target.goal.absolute_tolerance), sample_id must match, measurement revision must match observations.revision and captured_at_step must exceed last_mutation_step. If all hold, return supported and cite the measurement id. Valid fresh readback not matching the goal means contradicted; missing, unavailable, stale or mismatched evidence means unresolved with missing_evidence. Never assign the evaluator score.',
    'deliberation': 'Choose exactly one allowed route from the supplied summary. Prepare an existing intent, verify an existing fresh measurement, otherwise diagnose. Never execute tools.',
}


class DeviceProfile:
    prompts = PROMPTS
    projection = staticmethod(projection)

    @staticmethod
    def retrieve(role):
        return [{'id':'device-contract','source':'native session.describe + cases/devices task goal',
            'text':'Use the provided native input_schema and SI units. A control setpoint readback validates interface operation only. Scene geometry, physical acquisition and chemistry are outside this task.'}]

    @staticmethod
    def validate_role_contract(role, record, payload):
        if role == 'diagnostic' and record.intent:
            from jsonschema import Draft202012Validator
            cap = payload['capabilities'].get(record.intent.capability_id)
            if cap is None:
                raise ValueError('Use a listed capability')
            json_error = next(Draft202012Validator(cap['input_schema']).iter_errors(record.intent.arguments),None)
            if json_error:
                path = '.'.join(['arguments', *(str(p) for p in json_error.absolute_path)])
                raise ValueError('WORKFLOW_ARGUMENT_VALIDATION: ' + record.intent.capability_id
                                 + ' ' + path + ': ' + json_error.message)
        elif role == 'procedure':
            validate_role_contract(role,record,payload)


class ScriptedDeviceRoles:
    """Explicit integration-test double; never described as an expert model."""
    is_llm = False
    backend_id = 'scripted-native-device-test-double-v1'

    def __init__(self, task, slug):
        self.task, self.slug = deepcopy(task), slug
        self.actions = [(x['capability_id'],x['arguments']) for x in task['reference_probe']['steps']]
        self.index = 0

    def invoke(self,role,prompt,payload,schema):
        if role == 'diagnostic':
            obs = payload['observations']
            if not obs['inspection']:
                name,args = 'workflow.inspect',{}
            elif obs['last_mutation_step'] and not obs.get('measurement'):
                name,args = 'workflow.readback',{}
            elif self.index < len(self.actions):
                native,args = self.actions[self.index]
                name = self.slug+'::'+native
                self.index += 1
            else:
                name,args = 'workflow.readback',{}
            value = {'hypotheses':[{'cause':'control_goal','supporting_ids':[obs['inspection']['id']] if obs['inspection'] else [],'contradicting_ids':[]}],
                'intent':{'capability_id':name,'arguments':args,'discriminating_observation':'Read native public state after the command.','postcondition':'Public state reaches the device-control goal.'}}
        elif role == 'procedure':
            value = dict(payload['intent'],sample_id=payload['sample_id'],expected_revision=payload['expected_revision'],request_id=payload['request_id'])
            value.pop('discriminating_observation',None)
        elif role == 'verification':
            m = payload['observations']['measurement'];goal=payload['target']['goal']
            actual=m['native']['observations'][goal['state_id']]
            value={'status':'supported' if matches(actual['value'],goal['equals'],goal['absolute_tolerance']) else 'contradicted',
                'evidence_ids':[m['id']],'missing_evidence':[]}
        else:
            value={'backend_id':self.backend_id,'choice':'prepare_action' if payload['summary']['has_intent'] else 'verify' if payload['summary']['has_measurement'] else 'diagnose'}
        from app.models import OperationOptions
        if role == 'diagnostic' and schema is OperationOptions:
            from app.operation_decision import scripted_options
            return schema.model_validate(scripted_options(value, payload)).model_dump()
        return schema.model_validate(value).model_dump()
