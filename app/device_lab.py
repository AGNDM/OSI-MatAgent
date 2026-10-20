"""Native device tool adapter for resettable single-device control tasks."""
from copy import deepcopy
import json
import math

from jsonschema import Draft202012Validator

from app.device_registry import open_native, public_capabilities
from app.lab_adapter import GateRejected


def matches(value, expected, tolerance=1e-9):
    if isinstance(expected, bool):
        return type(value) is bool and value == expected
    if isinstance(expected, (int, float)):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and abs(value - expected) <= tolerance)
    return type(value) is type(expected) and value == expected


class DeviceLab:
    """Uses agent sessions only; developer fixture reference probes never enter state."""
    def __init__(self, spec, task):
        self.spec, self.task = deepcopy(spec), deepcopy(task)
        self._host, self._agent, self.instance_id, self.description = open_native(spec)
        self._native_caps = public_capabilities(self.description)
        self.sample_id = 'device:' + spec['slug']  # existing graph's identity channel
        self.revision = self.action_count = self.last_mutation_step = 0
        self.history, self._requests = [], {}
        self._inspection = self._measurement = None
        self._pending = None
        self._initial = self._read_native()

    def _key(self, cap):
        return self.spec['slug'] + '::' + cap

    def _read_native(self):
        # Project only public fields needed by this control task. Never read debug/evaluator.
        fields = [self.task['goal']['state_id']]
        if 'power.enabled' in self.description.get('state_slots', {}):
            fields.append('power.enabled')
        return self._agent.read(instance_id=self.instance_id, state_ids=list(dict.fromkeys(fields)))

    def target(self):
        return {'task':self.task['title'], 'scope':self.task['scope'], 'sample_id':self.sample_id,
            'asset_id':self.spec['asset_id'], 'instance_id':self.instance_id,
            'goal':deepcopy(self.task['goal']),
            'verification_rule':'After an accepted native action obtain workflow.readback. Compare the requested public state with goal.equals, respecting units, identity, quality and freshness. A written setpoint is not a completed physical experiment.'}

    def capabilities(self):
        caps = {
            'workflow.inspect':{'arguments':{},'input_schema':{'type':'object','properties':{},'additionalProperties':False},
                'source':'native agent-session read; no clock advance'},
            'workflow.readback':{'arguments':{},'input_schema':{'type':'object','properties':{},'additionalProperties':False},
                'source':'fresh native agent-session read; not a synthetic outcome'},
        }
        for name in self.task['tool_menu']:
            native = self._native_caps[name]
            caps[self._key(name)] = {'arguments':deepcopy(native['input_schema'].get('properties',{})),
                'input_schema':deepcopy(native['input_schema']), 'source':'native ChemLab '+self.spec['chem_id'],
                'description':native.get('function_zh',native.get('description','')),
                'example_arguments':deepcopy(native.get('example_arguments',{})),
                'execution':native.get('execution',{}).get('mode','unspecified'),
                'post_action':'On success adapter immediately reads public state; measurement holds that fresh readback. The read is counted separately.'}
        return caps

    def observe(self):
        return deepcopy({'revision':self.revision,'sample_id':self.sample_id,
            'inspection':self._inspection,'measurement':self._measurement,
            'last_mutation_step':self.last_mutation_step,
            'last_action':self.history[-1] if self.history else None})

    def validate(self, proposal, pending_operation=None):
        digest = json.dumps(proposal.model_dump(), sort_keys=True, allow_nan=False)
        if proposal.sample_id != self.sample_id:
            raise GateRejected('WRONG_DEVICE')
        if proposal.request_id in self._requests:
            if self._requests[proposal.request_id][0] != digest:
                raise GateRejected('REQUEST_CONFLICT')
            return
        if self._pending or pending_operation:
            raise GateRejected('PENDING_OPERATION_UNRESOLVED')
        if proposal.expected_revision != self.revision:
            raise GateRejected('STALE_PROPOSAL')
        cap = self.capabilities().get(proposal.capability_id)
        if cap is None:
            raise GateRejected('UNSUPPORTED_CAPABILITY')
        if list(Draft202012Validator(cap['input_schema']).iter_errors(proposal.arguments)):
            raise GateRejected('INVALID_ARGUMENT')
        if proposal.capability_id not in {'workflow.inspect','workflow.readback'} and self.last_mutation_step and self._measurement is None:
            raise GateRejected('READBACK_REQUIRED_AFTER_NATIVE_ACTION')
        if proposal.capability_id not in {'workflow.inspect','workflow.readback'} and not self._inspection:
            raise GateRejected('INSPECTION_REQUIRED')

    def _snapshot(self):
        native = self._read_native()
        return {'id':f'obs-{self.action_count}', 'sample_id':self.sample_id,'revision':self.revision,
            'captured_at_step':self.action_count,'quality':'valid' if native.get('status') == 'succeeded' else 'unavailable',
            'native':native}

    @staticmethod
    def action_cost(proposal):
        return 1 if proposal.capability_id in {'workflow.inspect','workflow.readback'} else 2

    def execute(self, proposal):
        self.validate(proposal)
        if proposal.request_id in self._requests:
            return deepcopy(self._requests[proposal.request_id][1])
        self.action_count += 1
        command_step = self.action_count
        cap = proposal.capability_id
        if cap == 'workflow.inspect':
            result = self._snapshot()
            self._inspection = result
        elif cap == 'workflow.readback':
            result = self._snapshot()
            self._measurement = result
        else:
            native = cap.split('::',1)[1]
            response = self._agent.invoke(instance_id=self.instance_id,capability_id=native,
                arguments=proposal.arguments,request_id=proposal.request_id)
            result = {'native_response':response}
            status = response.get('status')
            if status == 'succeeded':
                self.revision += 1
                self.last_mutation_step = self.action_count
                # Return actual post-command public evidence, not an old inspection.
                self.action_count += 1
                self._measurement = self._snapshot()
                result['post_action_readback'] = deepcopy(self._measurement)
            elif status in {'pending','running','accepted'}:
                self._pending = response.get('operation_id') or proposal.request_id
                result['error'] = {'code':'PENDING_OPERATION_UNRESOLVED', 'operation_id':self._pending}
            else:
                result['error'] = response.get('error') or {'code':'NATIVE_'+str(status)}
        event = {'step':command_step,'action_count_after':self.action_count,'proposal':proposal.model_dump(),'result':deepcopy(result),
            'revision':self.revision,'source':self.capabilities()[cap]['source']}
        self.history.append(event)
        self._requests[proposal.request_id] = (json.dumps(proposal.model_dump(),sort_keys=True,allow_nan=False),deepcopy(event))
        return event

    def score(self, report):
        # Independent re-read after the graph; no agent success flag is used as truth.
        final = self._read_native()
        goal = self.task['goal']
        obs = final['observations'][goal['state_id']]
        actual = obs.get('quality') in {'valid','estimated'} and matches(obs.get('value'),goal['equals'],goal['absolute_tolerance'])
        m = self._measurement
        cited = bool(m and m['id'] in report.get('evidence_ids',[]))
        fresh = bool(m and m['quality']=='valid' and m['revision']==self.revision
            and m['captured_at_step']>self.last_mutation_step)
        changed = not matches(self._initial['observations'][goal['state_id']]['value'],goal['equals'],goal['absolute_tolerance'])
        executed = any('native_response' in e['result'] and e['result']['native_response'].get('status')=='succeeded' for e in self.history)
        supported = report.get('status') == 'supported'
        return {'task_success':bool(actual and fresh and cited and executed and changed and supported),
            'native_goal_met':bool(actual),'fresh_readback':fresh,'evidence_cited':cited,
            'nontrivial_initial_state':changed,'accepted_native_action':executed, 'environment_setup':deepcopy(self._host._workflow_fixture_events),
            'false_success':bool(supported and not(actual and fresh and cited and executed and changed)),
            'final_public_read':final, 'scope':'control/tool-call task only'}
